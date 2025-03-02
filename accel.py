import os

import torch
from accelerate import Accelerator
from transformers import get_scheduler

from aux import SameLineLogger, log
#from modelops import load_data_state, load_loss_list, save_all_models
from modelops import save_all_models
from translate import encode


def chain_params(coupling_specs):
    for spec in coupling_specs:
        yield from spec.model.parameters()


class DataState:
    def __init__(self):
        self.batch_idx = -1
        self.epoch_idx = 0

    def is_continued(self):
        is_fresh = (self.batch_idx == -1) and (self.epoch_idx == 0)
        return not is_fresh

    def state_dict(self):
        return {'batch_idx': self.batch_idx, 'epoch_idx': self.epoch_idx}

    def load_state_dict(self, state_dict):
        self.batch_idx = state_dict['batch_idx']
        self.epoch_idx = state_dict['epoch_idx']

    def __str__(self):
        return 'DataState(batch_idx={}, epoch_idx={})'.format(self.batch_idx, self.epoch_idx)

    def __repr__(self):
        return self.__str__()


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

        self.train_loss_list = TrainLossList()
        self.data_state = DataState()

        self._init_acc_and_stuff()

    def _init_acc_and_stuff(self):
        self.accelerator = Accelerator(gradient_accumulation_steps=self.kwargs.accum_steps)

        epoch_len = len(self.train_set)
        train_len = epoch_len * self.kwargs.epochs

        num_warmup = int(train_len * 0.01)

        log(f"Warmup steps: {num_warmup}, epoch len: {epoch_len}, train len: {train_len}")

        optimizer = torch.optim.AdamW(chain_params(self.coupling_specs), lr=self.kwargs.lr)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup,
                                     num_training_steps=train_len * self.accelerator.num_processes)
        models = [s.model for s in self.coupling_specs]

        self.train_set, self.optimizer, self.lr_scheduler, *self.models = self.accelerator.prepare(
            self.train_set, optimizer, lr_scheduler, *models)

        self.accelerator.register_for_checkpointing(self.lr_scheduler, self.data_state, self.train_loss_list)

        if self.kwargs.continue_training:
            self.accelerator.load_state(self.kwargs.mdl_id)
            log(f"Reloaded data state: {self.data_state}")

    def train(self):
        self._main_loop()

        self.accelerator.wait_for_everyone()

        unwr_coupled_model = self.accelerator.unwrap_model(self.models[0])

        return unwr_coupled_model, self.train_loss_list

    def _prepare_inputs(self, batch_with_idxs):
        weird_inputs, src_k, tgt_k, _ = batch_with_idxs

        batch_size = weird_inputs['input_ids'].size()[0]

        proc_batch_size = batch_size / self.accelerator.num_processes

        from_proc_idx = int(self.accelerator.process_index * proc_batch_size)
        to_proc_idx = int((self.accelerator.process_index + 1) * proc_batch_size)

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

        _batch_idx = self.train_set.maybe_skip_ahead(self.data_state)

        print(f"Batching from idx {self.data_state.batch_idx + 1}")
        print(f"Starting from epoch {self.data_state.epoch_idx} and running up to {self.kwargs.epochs-1}")

        for _epoch_idx in range(self.data_state.epoch_idx, self.kwargs.epochs):
            for batch_with_bin_idxs in self.train_set:
                if _batch_idx > self.data_state.batch_idx:
                    inputs, src_k, tgt_k = self._prepare_inputs(batch_with_bin_idxs)

                    encoder_vecs = encode(self.models[src_k], inputs)
                    outputs = self.models[tgt_k](attention_mask=inputs['attention_mask'], labels=inputs['labels'], encoder_outputs=encoder_vecs)

                    loss = outputs.loss

                    self.train_loss_list.append(loss.item(), src_k, tgt_k)

                    self.accelerator.backward(loss)

                    self._step_and_perhaps_save(logger, _batch_idx, _epoch_idx, float(loss.item()))

                _batch_idx += 1

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

    def _step_and_perhaps_save(self, logger, batch_i, epoch_i, loss):
        self.optimizer.step()
        self.lr_scheduler.step()

        if self.accelerator.is_main_process and ((batch_i + 1) % self.kwargs.log_steps == 0):
            grad = self.get_total_grad()
            logger.step(batch_i, loss, self.lr_scheduler.get_last_lr()[0], grad)

        self.optimizer.zero_grad()

        if ((batch_i + 1) % self.kwargs.save_steps == 0) or ((batch_i + 1) % len(self.train_set) == 0):
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                logger.line_break()
                log(f"Saving at {batch_i + 1} steps, epoch {epoch_i + 1}")

                self._save_all(batch_i, epoch_i)

                logger.line_start()

    def _save_all(self, batch_i, epoch_i):
        epoch_len = len(self.train_set)

        ckpt_name = f"checkpoint-e{epoch_i + 1:02}-b{batch_i + 1:07}" if (
                    (batch_i + 1) % epoch_len) else f"checkpoint-e{epoch_i + 1}-full"

        this_location = os.path.join(self.kwargs.save_location, ckpt_name)
        if os.path.exists(this_location):
            raise FileExistsError("Cannot overwrite existing checkpoint")

        self.data_state.epoch_idx = epoch_i
        self.data_state.batch_idx = batch_i
        log(f"Saving data state as epoch={epoch_i}, batch={batch_i}")

        self.accelerator.save_state(this_location)

        model_to_save = self.accelerator.unwrap_model(self.models[0])

        save_all_models(this_location, model_to_save, self.coupling_specs[0].tokenizer,
                        self.coupling_specs, self.train_loss_list, trainer=None,
                        data_state=(batch_i, epoch_i))
