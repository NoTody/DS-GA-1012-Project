import os
import sys
import torch
import copy
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from data_utils import *
from model_utils import *

class CRTransformer(LightningModule):
    def __init__(self, args):

        super().__init__(hparams)
        # Set our init args as class attributes
        self.hparams.update(vars(hparams))
        
        # Build models
        __build_model()

        # Define metrics
        self.accuracy = Accuracy()
        self.f1 = F1Score()
    
    #############################
    # Training / Validation HOOKS
    ############################# 
    def __build_model():
        # Define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name, 
                                                       use_fast=True)

        # Define config
        self.config = AutoConfig.from_pretrained(self.hparams.model_name, 
                                                 num_labels=self.hparams.num_labels,
                                                 output_hidden_states=True)
        # Define model
        self.student = model_init(self.hparams.model_name, self.config)
        self.teacher = copy.deepcopy(self.student)
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def _EMA_update(self, it):
        """
        Exponential Moving Average update of the student
        """
        m = self.momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    #############################
    # Training / Validation HOOKS
    #############################

    def forward(self, **inputs):
        return self.model(**inputs)

    def forward_one_epoch(self, batch, batch_idx):
        b_input_ids, b_attn_mask, b_labels = batch['ori']['input_ids'], batch['ori']['attention_mask'], batch['ori']['labels']
        outputs_ori = self.student(b_input_ids, b_attn_mask)
        
        b_input_ids, b_attn_mask, b_labels = batch['str_adv']['input_ids'], batch['str_adv']['attention_mask'], batch['str_adv']['labels']
        outputs_str_adv = self.teacher(b_input_ids, b_attn_mask)
        
        b_input_ids, b_attn_mask, b_labels = batch['weak_aug']['input_ids'], batch['weak_aug']['attention_mask'], batch['weak_aug']['labels']
        outputs_weak_aug = self.student(b_input_ids, b_attn_mask)
        
        hidden_states_weak_aug = outputs_weak_aug.hidden_states
        hidden_states_str_adv = outputs_str_adv.hidden_states
        
        criterion = nn.CrossEntropyLoss()
        # calcualate self-supervised loss from cls embeddings
        for i in range(self.top_k_layers - 1):
            cur_hidden = hidden_states_weak_aug[i][:, 0, :]
            y_aug = torch.stack([y, cur_hidden], dim=1)
            cur_hidden = hidden_states_str_adv[i][:, 0, :]
            y_adv = torch.stack([y, cur_hidden], dim=1)
        # layernorm
        y_aug = F.layer_norm(y_aug, y_aug.shape[1:])
        y_adv = F.layer_norm(y_adv, y_adv.shape[1:])
        # calculate smooth l1 loss
        sz = y_aug.size(-1)
        loss_scale = 1 / math.sqrt(sz)
        selfsup_loss = loss_scale * F.smooth_l1_loss(y_aug.float(), y_adv.float(), reduction="none", 
                                                     beta=self.loss_beta).sum(dim=-1).sum()
        # calculate supervised loss from true label
        criterion = nn.CrossEntropyLoss()
        logits = outputs_ori.logits
        sup_loss = criterion(logits, b_labels)
        # calculate final loss
        loss = (1 - self.hparams.lambda) * selfsup_loss + self.hparams.lambda * sup_loss
        # get predict
        preds = torch.argmax(logits, dim=1)
        return {'loss': loss, 'preds': preds, 'labels': b_labels}
    
    def training_step(self, batch, batch_idx):
        forward_outputs = self.forward_one_epoch(batch, batch_idx)
        train_loss = forward_outputs['loss']
        b_input_ids = forward_outputs['input_ids']
        # Tensorboard logging for model graph and loss
        #self.logger.experiment.add_graph(self.model, input_to_model=b_input_ids, verbose=False, use_strict_trace=True)
        #self.logger.experiment.add_scalars('loss', {'train_loss': train_loss}, self.global_step)
        self.log("train_loss", train_loss, on_epoch=False, on_step=True, prog_bar=True)
        return train_loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # update student/teacher with EMA after each batch
        it = self.global_step
        self._EMA_update(it)
        
    def validation_step(self, batch, batch_idx):
        forward_outputs = self.forward_one_epoch(batch, batch_idx)
        val_loss = forward_outputs['loss']
        preds = forward_outputs['preds']
        labels = forward_outputs['labels']
        self.accuracy(preds, labels)
        self.f1(preds, labels)
        # Calling self.log will surface up scalars for you in TensorBoard
        #self.logger.experiment.add_scalars('loss', {'val_loss': val_loss}, self.global_step)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val_acc", self.accuracy, on_epoch=True, on_step=False, prog_bar=True)
        # self.log("val_f1", self.f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_acc", self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.f1, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        forward_outputs = self.forward_one_epoch(batch, batch_idx)
        preds = forward_outputs['preds']
        b_labels = forward_outputs['labels']
        test_loss = forward_outputs['loss']
        #cls_hidden_states = forward_outputs['hidden_states'][0][:, 0, :]
        # Reuse the validation_step for testing
        # Visualize dimensionality reduced labels
        # print(cls_hidden_states.shape)
        # print(b_labels.shape)
        #self.logger.experiment.add_embedding(cls_hidden_states, metadata=b_labels.tolist(), global_step=self.global_step)
        self.accuracy(preds, b_labels)
        self.f1(preds, b_labels)
        self.log("test_acc", self.accuracy)
        self.log("test_f1", self.f1)
        return test_loss

    def configure_optimizers(self):
        # set no decay for bias and normalziation weights
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # define optimizer / scheduler
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        self.warmup_steps = 0.06 * self.total_steps
        if self.hparams.scheduler_name == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.hparams.scheduler_name == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
            )
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    ####################
    # DATA RELATED HOOKS
    ####################
    def setup(self, stage=None):
      # dataset setup
      if stage == "fit" or stage is None:
        # train dataset assign
        train_path_ori = "./datasets/" + self.hparams.dataset_name_ori + "_train.csv"
        train_path_str_adv = "./datasets/" + self.hparams.dataset_name_str_adv + "_train.csv"
        train_path_weak_aug = "./datasets/" + self.hparams.dataset_name_weak_aug + "_train.csv"
        # read/generate three ways dataset
        df_train_ori = pd.read_csv(train_path_ori)
        df_train_weak_aug = pd.read_csv(train_path_weak_aug)
        df_train_str_adv = pd.read_csv(train_path_str_adv)
        
        self.ds_train_ori = SequenceDataset(df_train_ori, self.dataset_name, self.tokenizer,
                                            max_seq_length=self.max_seq_length)
        self.ds_train_weak_aug = SequenceDataset(df_train_weak_aug, self.dataset_name, self.tokenizer,
                                                 max_seq_length=self.max_seq_length)
        self.ds_train_str_adv = SequenceDataset(df_train_str_adv, self.dataset_name, self.tokenizer,
                                                max_seq_length=self.max_seq_length)
        
        # val dataset assign
        val_path_ori = "./datasets/" + self.hparams.dataset_name_ori + "_val.csv"
        val_path_str_adv = "./datasets/" + self.hparams.dataset_name_str_adv + "_val.csv"
        val_path_weak_aug = "./datasets/" + self.hparams.dataset_name_weak_aug + "_val.csv"
        df_val_ori = pd.read_csv(val_path_ori)
        df_val_weak_aug = pd.read_csv(val_path_weak_aug)
        df_val_str_adv = pd.read_csv(val_path_str_adv)
        self.ds_val_ori = SequenceDataset(df_val_ori, self.dataset_name, self.tokenizer, 
                                          max_seq_length=self.max_seq_length)
        self.ds_val_weak_aug = SequenceDataset(df_val_weak_aug, self.dataset_name, self.tokenizer, 
                                               max_seq_length=self.max_seq_length)
        self.ds_val_str_adv = SequenceDataset(df_val_str_adv, self.dataset_name, self.tokenizer, 
                                              max_seq_length=self.max_seq_length)
        # Calculate total steps
        tb_size = self.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(df_train) // tb_size) // ab_size
        print(f"total step: {self.total_steps}")
        
    if stage == "test" or stage is None:
        # test dataset assign
        test_path = "./datasets/" + self.testset_name + "_test.csv"
        df_test = pd.read_csv(test_path)
        self.ds_test = SequenceDataset(df_test, self.testset_name, self.tokenizer, max_seq_length=self.max_seq_length)

    def train_dataloader(self):
        self.ds_train_ori = DataLoader(self.ds_train_ori, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        ds_train_weak_aug = DataLoader(self.ds_train_weak_aug, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        ds_train_str_adv = DataLoader(self.ds_train_str_adv, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = model_utils.cosine_scheduler(self.hparams.momentum_teacher, 1,
                                                              self.hparams.max_epochs, len(self.ds_train_ori))
        
        return {"ori": self.ds_train_ori, "weak_aug": ds_train_weak_aug, "str_adv": ds_train_str_adv}
    
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        # config parameters
        parser.add_argument("--accumulate_grad_batches", type=int, default=1)
        parser.add_argument("--model_name", type=str, default="bert-base-uncased")
        parser.add_argument("--dataset_name_ori", type=str, default="agnews")
        parser.add_argument("--dataset_name_str_adv", type=str, default="agnews")
        parser.add_argument("--dataset_name_weak_aug", type=str, default="agnews")
        parser.add_argument("--testset_name", type=str, default="agnews")
        parser.add_argument("--num_workers", type=int, default=10)
        parser.add_argument("--max_epochs", type=int, default=5)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--max_seq_length", type=int, default=256)
        parser.add_argument("--mode", type=str, default="train")
        parser.add_argument("--load_path", type=str, default=None)
        # model parameters
        parser.add_argument("--lr", type=float, default=2e-5)
        parser.add_argument("--num_labels", type=int, default=4)
        parser.add_argument("--lambda", type=float, default=0.5)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument('--momentum_teacher', default=0.996, type=float)
        parser.add_argument("--scheduler_name", type=str, default="cosine")
