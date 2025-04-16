import os

import torch

from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import get_scheduler

from aux import SameLineLogger, log
from data import DataState
from langconv import is_dec_only_llm
from modelops import save_all_models, report_devices
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

    def _split_batch_and_bin_idxs(self, batch_with_idxs):
        if self.is_generative:
            batch, _ = batch_with_idxs
            src_k = 0
            tgt_k = 0
        else:
            batch, src_k, tgt_k, _ = batch_with_idxs
        return batch, src_k, tgt_k

    def _prepare_inputs(self, batch, sub_batch_idx, sub_batch_size, proc_batch_size):
        from_proc_idx = proc_batch_size * self.accelerator.process_index + sub_batch_size * sub_batch_idx
        to_proc_idx = from_proc_idx + sub_batch_size

        #log(f"----> DEBUG for sub_b idx {sub_batch_idx}, proc {self.accelerator.process_index}: {from_proc_idx}:{to_proc_idx}")

        return {k: batch[k][from_proc_idx:to_proc_idx].to(self.accelerator.device) for k in batch}

    def _get_split_batch_params(self, batch):
        batch_nr_snts = batch['input_ids'].size()[0]
        snt_nr_words = batch['input_ids'].size()[1]

        assert batch_nr_snts % self.accelerator.num_processes == 0, "Batch size must be divisible by number of processes."

        proc_batch_nr_snts = batch_nr_snts // self.accelerator.num_processes

        if self.kwargs.nr_snts_in_batch > 0:
            sub_batch_size = self.kwargs.nr_snts_in_batch
        else:
            sub_batch_size = max(1, self.kwargs.nr_words_in_batch // snt_nr_words)
        #log(f"DEBUG: #words/snt {snt_nr_words} X #snt in sub batch {sub_batch_size} = {snt_nr_words*sub_batch_size} ~ {self.kwargs.nr_words_in_batch}", accelerator=self.accelerator)

        nr_steps = -(proc_batch_nr_snts // -sub_batch_size)

        #log(f"--> DEBUG: sub_batch {sub_batch_size} X steps {nr_steps} ~ {proc_batch_nr_snts} ({batch_nr_snts} / {self.accelerator.num_processes})", accelerator=self.accelerator)
        return sub_batch_size, nr_steps, proc_batch_nr_snts

    def _main_loop(self):
        #countdown_till_do_it_once = 0

        if self.accelerator.is_main_process:
            logger = SameLineLogger(len(self.train_set), self.kwargs.epochs)
            logger.line_start()
        else:
            logger = None

        self.models[0].train()
        self.train_set.thats_where(self.data_state)

        for _epoch_idx in range(self.data_state.epoch_idx, self.kwargs.epochs):
            for batch_with_bin_idxs, epoch_batch_idx in self.train_set:
                batch, src_k, tgt_k = self._split_batch_and_bin_idxs(batch_with_bin_idxs)
                sub_batch_size, nr_steps, proc_batch_size = self._get_split_batch_params(batch)

                loss = None

                for sub_batch_idx in range(nr_steps):
                    inputs = self._prepare_inputs(batch, sub_batch_idx, sub_batch_size, proc_batch_size)

                    if self.is_generative:
                        inputs['labels'] = inputs['input_ids']
                        outputs = self.models[0](**inputs)
                    else:
                        encoder_vecs = encode(self.models[src_k], inputs)
                        outputs = self.models[tgt_k](attention_mask=inputs['attention_mask'], labels=inputs['labels'], encoder_outputs=encoder_vecs)

                    loss = outputs.loss

                    #if countdown_till_do_it_once > 0:
                    #    countdown_till_do_it_once -= 1
                    #elif countdown_till_do_it_once == 0:
                    if sub_batch_idx == 5:
                        batch_size = sum([inputs[k].size()[0] * inputs[k].size()[1] for k in 'input_ids labels attention_mask'.split(' ')])
                        report_devices(f"training memory usage (batch size: {batch_size}; inputs:" +
                                       f"snts {inputs['input_ids'].size()[0]} X words {inputs['input_ids'].size()[1]})",
                                       self.accelerator, self.models[0])
                        countdown_till_do_it_once = 0

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
