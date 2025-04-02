import os

import torch

from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import get_scheduler

from aux import SameLineLogger, log
from data import DataState
from langconv import is_dec_only_llm
from modelops import save_all_models
from translate import encode


def chain_params(coupling_specs):
    for spec in coupling_specs:
        yield from spec.model.parameters()


class TrainLossList:
    def __init__(self):
        self.data = []

    def append(self, loss_val, src_k, tgt_k):
        self.data.append((loss_val, src_k, tgt_k))

    def state_dict(self):
        return {'data': self.data}

    def load_state_dict(self, state_dict):
        self.data = state_dict['data']



class SwitchingAccelerator:
    def __init__(self, coupling_specs, train_set, train_kwargs):
        self.coupling_specs = coupling_specs

        self.train_set = train_set
        self.kwargs = train_kwargs

        self.is_generative = is_dec_only_llm(self.coupling_specs[0].tokenizer)

        self.train_loss_list = TrainLossList()
        self.data_state = DataState(epoch_idx=0)

        self._init_acc_and_stuff()

    def _init_acc_and_stuff(self):
        #self.accelerator = Accelerator(gradient_accumulation_steps=self.kwargs.accum_steps, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        #self.accelerator = Accelerator(gradient_accumulation_steps=self.kwargs.accum_steps)
        self.accelerator = Accelerator()

        epoch_len = len(self.train_set)
        train_len = epoch_len * self.kwargs.epochs

        num_warmup = int(train_len * 0.01)

        log(f"Warmup steps: {num_warmup}, epoch len: {epoch_len}, train len: {train_len}", accelerator=self.accelerator)

        opt = torch.optim.AdamW(chain_params(self.coupling_specs), lr=self.kwargs.lr)
        lr_scheduler = get_scheduler("linear", optimizer=opt, num_warmup_steps=num_warmup,
                                     num_training_steps=train_len * self.accelerator.num_processes)
        models = [s.model for s in self.coupling_specs]

        self.optimizer, self.lr_scheduler, *self.models = self.accelerator.prepare(opt, lr_scheduler, *models)

        self.accelerator.register_for_checkpointing(self.lr_scheduler, self.data_state, self.train_loss_list)

        if self.kwargs.continue_training:
            self.accelerator.load_state(self.kwargs.mdl_id)
            log(f"Reloaded data state: {self.data_state}", accelerator=self.accelerator)

    def train(self):
        try:
            self._main_loop()
        except Exception as e:
            #in multi-process scenarios it is hard to read the stack trace, so just show one:
            if self.accelerator.is_main_process:
                raise e

        self.accelerator.wait_for_everyone()

        unwr_coupled_model = self.accelerator.unwrap_model(self.models[0])

        return unwr_coupled_model, self.train_loss_list

    def _prepare_inputs(self, batch_with_idxs, accum_idx):
        if self.is_generative:
            weird_inputs, _ = batch_with_idxs
            src_k = 0
            tgt_k = 0
        else:
            weird_inputs, src_k, tgt_k, _ = batch_with_idxs

        batch_size = weird_inputs['input_ids'].size()[0]

        split_into = self.accelerator.num_processes * self.kwargs.accum_steps

        assert batch_size % split_into == 0, "Batch size must be divisible by number of processes X accumulation steps."

        proc_batch_size = batch_size / split_into

        from_proc_idx = int((accum_idx * self.accelerator.num_processes + self.accelerator.process_index) * proc_batch_size)
        to_proc_idx = int((accum_idx * self.accelerator.num_processes + self.accelerator.process_index + 1) * proc_batch_size)

        unweird_inputs = {k: weird_inputs[k][from_proc_idx:to_proc_idx].to(self.accelerator.device)
                          for k in weird_inputs}

        return unweird_inputs, src_k, tgt_k

    def _main_loop(self):
        if self.accelerator.is_main_process:
            logger = SameLineLogger(len(self.train_set), self.kwargs.epochs)
            logger.line_start()
        else:
            logger = None

        self.models[0].train()

        #£_batch_idx, skipped = self.train_set.maybe_skip_ahead(self.data_state)
        #print(self.data_state, _batch_idx, skipped)
        self.train_set.thats_where(self.data_state)

        for _epoch_idx in range(self.data_state.epoch_idx, self.kwargs.epochs):
            for batch_with_bin_idxs, epoch_batch_idx in self.train_set:
                for accum_idx in range(self.kwargs.accum_steps):
                    inputs, src_k, tgt_k = self._prepare_inputs(batch_with_bin_idxs, accum_idx)

                    if self.is_generative:
                        inputs['labels'] = inputs['input_ids']
                        outputs = self.models[0](**inputs)
                    else:
                        encoder_vecs = encode(self.models[src_k], inputs)
                        outputs = self.models[tgt_k](attention_mask=inputs['attention_mask'], labels=inputs['labels'], encoder_outputs=encoder_vecs)

                    loss = outputs.loss

                    self.train_loss_list.append(loss.item(), src_k, tgt_k)

                    self.accelerator.backward(loss)

                self._step_and_perhaps_save(logger, epoch_batch_idx, _epoch_idx, float(loss.item()))

        if self.accelerator.is_main_process:
            logger.line_break()

    def get_total_grad(self):
        result = 0
        grad_count = 0
        all_count = 0

        for p in self.models[0].parameters():
            if p.grad is not None:
                result += p.grad.abs().mean().item()
                grad_count += 1
            all_count += 1

        return result/grad_count if grad_count > 0 else -1

    def _step_and_perhaps_save(self, logger, epoch_batch_idx, epoch_i, loss):
        epoch_len = len(self.train_set)
        global_batch_idx = epoch_batch_idx + epoch_i * epoch_len

        self.optimizer.step()
        self.lr_scheduler.step()

        is_end_of_epoch = (epoch_batch_idx == epoch_len)

        if self.accelerator.is_main_process and (epoch_batch_idx % self.kwargs.log_steps == 0 or is_end_of_epoch):
            grad = self.get_total_grad()
            logger.step(global_batch_idx, epoch_batch_idx, epoch_i, loss, self.lr_scheduler.get_last_lr()[0], grad)

        self.optimizer.zero_grad()

        if (global_batch_idx % self.kwargs.save_steps == 0) or is_end_of_epoch:
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                logger.line_break()
                log(f"Saving at {epoch_batch_idx} steps, epoch {epoch_i + 1} ({global_batch_idx} global steps)", accelerator=self.accelerator)

                self._save_all(global_batch_idx, epoch_i)

                logger.line_start()

    def _save_all(self, global_batch_idx, epoch_i):
        epoch_len = len(self.train_set)

        ckpt_name = (f"checkpoint-e{epoch_i + 1:02}-" +
                     (f"b{global_batch_idx:07}" if (global_batch_idx % epoch_len) else f"full"))

        this_location = os.path.join(self.kwargs.save_location, ckpt_name)
        if os.path.exists(this_location):
            raise FileExistsError(f"Cannot overwrite existing checkpoint {this_location}!")

        self.data_state.copy_from(self.train_set.where_are_we(), epoch_idx=epoch_i)

        model_to_save = self.accelerator.unwrap_model(self.models[0])

        save_all_models(this_location, model_to_save, self.coupling_specs[0].tokenizer,
                        self.coupling_specs, trainer=self.accelerator)
