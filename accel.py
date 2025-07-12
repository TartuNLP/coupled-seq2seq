import os

import torch
import math

from accelerate import Accelerator, DistributedDataParallelKwargs
from sympy.matrices.kind import num_mat_mul
from transformers import get_scheduler

from aux import SameLineLogger, log
from data import DataState, BatchingIterator
from modelops import save_all_models, report_devices


def chain_params(coupling_specs):
    for spec in coupling_specs:
        yield from spec.model.parameters()


class TrainLossList:
    def __init__(self):
        self.data = []

    def append(self, loss_val, sub_batch_idx, epoch_batch_idx, _epoch_idx):
        self.data.append((loss_val, sub_batch_idx, epoch_batch_idx, _epoch_idx))

    def state_dict(self):
        return {'data': self.data}

    def load_state_dict(self, state_dict):
        self.data = state_dict['data']



class SwitchingAccelerator:
    def __init__(self, train_set, train_kwargs, model, tokenizer):
        self.kwargs = train_kwargs
        self.train_set_iter = BatchingIterator(train_set, self.kwargs.batch_size, tokenizer)

        self.model = model
        self.tokenizer = tokenizer

        self.train_loss_list = TrainLossList()
        self.data_state = DataState(epoch_idx=0)

        self._init_acc_and_stuff()

    def __handle_accum(self):

        assert self.kwargs.batch_size % (self.accelerator.num_processes * self.kwargs.nr_sents_per_gpu) == 0,\
            "batch size must be divisible by number of processes and number of segments per GPU"

        accum_steps = int((self.kwargs.batch_size / self.accelerator.num_processes) / self.kwargs.nr_sents_per_gpu)
        self.accelerator.gradient_accumulation_steps = accum_steps

        log(f"Nr sents/GPU: {self.kwargs.nr_sents_per_gpu}, accum steps: {accum_steps}, " +
            f"nr. procs: {self.accelerator.num_processes}, batch size: {self.kwargs.batch_size}")

    def ___get_train_scalars(self):
        epoch_len = len(self.train_set_iter)
        train_len = epoch_len * self.kwargs.epochs

        num_warmup = int(train_len * 0.01)

        log(f"Warmup steps: {num_warmup}, epoch len: {epoch_len}, train len: {train_len}", accelerator=self.accelerator)

        return train_len, num_warmup

    def __init_opt_lr_and_what_else(self):
        train_len, num_warmup = self.___get_train_scalars()

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.kwargs.lr)
        lr_scheduler = get_scheduler("linear", optimizer=opt, num_warmup_steps=num_warmup,
                                     num_training_steps=train_len * self.accelerator.num_processes)

        self.optimizer, self.lr_scheduler, self.model = self.accelerator.prepare(opt, lr_scheduler, self.model)

        self.accelerator.register_for_checkpointing(self.lr_scheduler, self.data_state, self.train_loss_list)

    def _init_acc_and_stuff(self):
        #self.accelerator = Accelerator(gradient_accumulation_steps=self.kwargs.accum_steps, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

        self.accelerator = Accelerator()

        self.__handle_accum()

        self.__init_opt_lr_and_what_else()

        if self.kwargs.continue_training:
            self.accelerator.load_state(self.kwargs.mdl_id)
            log(f"Reloaded data state: {self.data_state}", accelerator=self.accelerator)

    def train(self):
        try:
            self._main_loop()
        except Exception as e:
            #in multiprocess scenarios it is hard to read the stack trace, so just show one:
            if self.accelerator.is_main_process:
                raise e

        self.accelerator.wait_for_everyone()

        unwr_coupled_model = self.accelerator.unwrap_model(self.model)

        return unwr_coupled_model

    def _prepare_inputs(self, batch, sub_batch_idx, sub_batch_size, proc_batch_size):
        from_proc_idx = proc_batch_size * self.accelerator.process_index + sub_batch_size * sub_batch_idx
        to_proc_idx = from_proc_idx + sub_batch_size

        #log(f"----> DEBUG for sub_b idx {sub_batch_idx}, proc {self.accelerator.process_index}: {from_proc_idx}:{to_proc_idx}")

        return {k: batch[k][from_proc_idx:to_proc_idx].to(self.accelerator.device) for k in batch}

    def _get_split_batch_params(self, batch):
        batch_nr_snts = self.kwargs.batch_size

        assert batch_nr_snts % self.accelerator.num_processes == 0, "Batch size must be divisible by number of processes."

        proc_batch_nr_snts = batch_nr_snts // self.accelerator.num_processes

        sub_batch_size = self.kwargs.nr_sents_per_gpu

        nr_steps = -(proc_batch_nr_snts // -sub_batch_size)

        #log(f"--> DEBUG: sub_batch {sub_batch_size} X steps {nr_steps} ~ {proc_batch_nr_snts} ({batch_nr_snts} / {self.accelerator.num_processes})", accelerator=self.accelerator)
        return sub_batch_size, nr_steps, proc_batch_nr_snts

    def _main_loop(self):
        if self.accelerator.is_main_process:
            logger = SameLineLogger(len(self.train_set_iter), self.kwargs.epochs)
            logger.line_start()
        else:
            logger = None

        self.model.train()
        self.train_set_iter.thats_where(self.data_state)

        for _epoch_idx in range(self.data_state.epoch_idx, self.kwargs.epochs):
            for batch, epoch_batch_idx in self.train_set_iter:
                sub_batch_size, nr_steps, proc_batch_size = self._get_split_batch_params(batch)

                with self.accelerator.accumulate(self.model):
                    loss = None
                    for sub_batch_idx in range(nr_steps):
                        inputs = self._prepare_inputs(batch, sub_batch_idx, sub_batch_size, proc_batch_size)

                        inputs['labels'] = inputs['input_ids']
                        outputs = self.model(**inputs)

                        loss = outputs.loss

                        if epoch_batch_idx % 5 == 0 and sub_batch_idx == 0:
                            batch_size = sum([inputs[k].size()[0] * inputs[k].size()[1] for k in 'input_ids labels attention_mask'.split(' ')])
                            report_devices(f"training memory usage (batch size: {batch_size}; inputs:" +
                                           f"snts {inputs['input_ids'].size()[0]} X words {inputs['input_ids'].size()[1]})",
                                           self.accelerator, self.model)

                        self.train_loss_list.append(loss.item(), sub_batch_idx, epoch_batch_idx, _epoch_idx)

                        self.accelerator.backward(loss)

                #assert self.accelerator.sync_gradients, "It is not time to sync gradients yet."
                self._step_and_perhaps_save(logger, epoch_batch_idx, _epoch_idx, float(loss.item()))

        if self.accelerator.is_main_process:
            logger.line_break()

    def get_total_grad(self):
        result = 0
        grad_count = 0
        all_count = 0

        for p in self.model.parameters():
            if p.grad is not None:
                result += p.grad.abs().mean().item()
                grad_count += 1
            all_count += 1

        return result/grad_count if grad_count > 0 else -1

    def _step_and_perhaps_save(self, logger, epoch_batch_idx, epoch_i, loss):
        epoch_len = len(self.train_set_iter)
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
        epoch_len = len(self.train_set_iter)

        ckpt_name = (f"checkpoint-e{epoch_i + 1:02}-" +
                     (f"b{global_batch_idx:07}" if (global_batch_idx % epoch_len) else f"full"))

        this_location = os.path.join(self.kwargs.save_location, ckpt_name)
        if os.path.exists(this_location):
            raise FileExistsError(f"Cannot overwrite existing checkpoint {this_location}!")

        self.data_state.copy_from(self.train_set_iter.where_are_we(), epoch_idx=epoch_i)

        model_to_save = self.accelerator.unwrap_model(self.model)

        save_all_models(this_location, model_to_save, self.tokenizer, trainer=self.accelerator)
