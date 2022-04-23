import os
import sys
import torch
import copy
import argparse
import math
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.trainer.supporters import CombinedLoader
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
    def __init__(self, hparams):

        super().__init__()
        # Set our init args as class attributes
        self.hparams.update(vars(hparams))
        
        # Build models
        self.__build_model()

        # Define metrics
        self.accuracy = Accuracy()
        self.accuracy_ori = Accuracy()
        self.accuracy_ssmba = Accuracy()
        self.accuracy_eda = Accuracy()
        self.accuracy_tf = Accuracy()

        self.f1 = F1Score()
    
    #############################
    # Training / Validation HOOKS
    ############################# 
    def __build_model(self):
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
        #print(len(self.ds_train_ori)) 
        #print(len(self.momentum_schedule))
        #print(self.momentum_schedule[34380])
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    #############################
    # Training / Validation HOOKS
    #############################

    def forward(self, **inputs):
        return self.student(**inputs)

    def forward_one_epoch(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch['ori']['input_ids'], batch['ori']['attention_mask'], batch['ori']['labels']
        #print(f"input_ids: {input_ids.shape}")
        outputs_ori = self.student(input_ids, attn_mask)
        
        input_ids, attn_mask, labels = batch['str_adv']['input_ids'], batch['str_adv']['attention_mask'], batch['str_adv']['labels']
        outputs_str_adv = self.teacher(input_ids, attn_mask)
        
        input_ids, attn_mask, labels = batch['weak_aug']['input_ids'], batch['weak_aug']['attention_mask'], batch['weak_aug']['labels']
        outputs_weak_aug = self.student(input_ids, attn_mask)
        
        hidden_states_weak_aug = outputs_weak_aug.hidden_states
        hidden_states_str_adv = outputs_str_adv.hidden_states
        
        if self.hparams.loss_func == 'l1_smooth': 
            criterion = nn.CrossEntropyLoss()
            # calcualate self-supervised loss from cls embeddings
            for i in range(self.hparams.top_k_layers - 1):
                cur_hidden = hidden_states_weak_aug[i][:, 0, :]
                cur_hidden = torch.unsqueeze(cur_hidden, 1) 
                if i == 0:
                    y_aug = cur_hidden
                else:
                    y_aug = torch.cat((y_aug, cur_hidden), dim=1)
                
                cur_hidden = hidden_states_str_adv[i][:, 0, :]
                cur_hidden = torch.unsqueeze(cur_hidden, 1) 
                if i == 0:
                    y_adv = cur_hidden
                else:
                    y_adv = torch.cat((y_adv, cur_hidden), dim=1)
            
            # layernorm
            y_aug = F.layer_norm(y_aug, y_aug.shape[1:])
            y_adv = F.layer_norm(y_adv, y_adv.shape[1:])
            #print(f"y_aug: {y_aug.shape}")
            #print(f"y_aug: {y_adv.shape}")
            # calculate smooth l1 loss
            sz = y_aug.size(-1)
            loss_scale = 1 / math.sqrt(sz)
            selfsup_loss = loss_scale * F.smooth_l1_loss(y_aug.float(), y_adv.float(), reduction="none", beta=self.hparams.loss_beta).sum(dim=-1).sum()
            # calculate supervised loss from true label
            criterion = nn.CrossEntropyLoss()
            logits = outputs_ori.logits
            sup_loss = criterion(logits, labels)
            # calculate final loss
            loss = (1 - self.hparams.lamb) * selfsup_loss + self.hparams.lamb * sup_loss
            # get predict
            preds = torch.argmax(logits, dim=1)
        else:
            raise NotImplementedError()

        return {'loss': loss, 'preds': preds, 'labels': labels}
   
    def evaluate(self, batch, stage=None):
        input_ids, attn_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels'] 
        outputs = self.student(input_ids, attn_mask)
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, labels)
        if stage:
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc", self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
    def _test(self, batch, stage=None):
        input_ids, attn_mask, labels = batch['ori']['input_ids'], batch['ori']['attention_mask'], batch['ori']['labels']
        #print(f"input_ids: {input_ids.shape}")
        outputs_ori = self.student(input_ids, attn_mask)
        logits = outputs_ori.logits
        hidden_states = outputs_ori.hidden_states
        criterion = nn.CrossEntropyLoss()
        loss_ori = criterion(logits, labels)
        preds_ori = torch.argmax(logits, dim=1)
        self.accuracy_ori(preds_ori, labels)
        if stage:
            self.log(f"{stage}_loss_ori", loss_ori, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc_ori", self.accuracy_ori, on_step=False, on_epoch=True, prog_bar=True)
 
        input_ids, attn_mask, labels = batch['ssmba']['input_ids'], batch['ssmba']['attention_mask'], batch['ssmba']['labels']
        outputs_ssmba = self.student(input_ids, attn_mask)
        logits = outputs_ssmba.logits
        hidden_states = outputs_ssmba.hidden_states
        criterion = nn.CrossEntropyLoss()
        loss_ssmba = criterion(logits, labels)
        preds_ssmba = torch.argmax(logits, dim=1)
        self.accuracy_ssmba(preds_ssmba, labels)
        if stage:
            self.log(f"{stage}_loss_ssmba", loss_ssmba, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc_ssmba", self.accuracy_ssmba, on_step=False, on_epoch=True, prog_bar=True)
        
        input_ids, attn_mask, labels = batch['eda']['input_ids'], batch['eda']['attention_mask'], batch['eda']['labels']
        outputs_eda = self.student(input_ids, attn_mask)
        logits = outputs_eda.logits
        hidden_states = outputs_eda.hidden_states
        criterion = nn.CrossEntropyLoss()
        loss_eda = criterion(logits, labels)
        preds_eda = torch.argmax(logits, dim=1)
        self.accuracy_eda(preds_eda, labels)
        if stage:
            self.log(f"{stage}_loss_eda", loss_eda, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc_eda", self.accuracy_eda, on_step=False, on_epoch=True, prog_bar=True)
        
        input_ids, attn_mask, labels = batch['tf']['input_ids'], batch['tf']['attention_mask'], batch['tf']['labels']
        outputs_tf = self.student(input_ids, attn_mask)
        logits = outputs_tf.logits
        hidden_states = outputs_tf.hidden_states
        criterion = nn.CrossEntropyLoss()
        loss_tf = criterion(logits, labels)
        preds_tf = torch.argmax(logits, dim=1)
        self.accuracy_tf(preds_tf, labels)
        if stage:
            self.log(f"{stage}_loss_tf", loss_tf, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc_tf", self.accuracy_tf, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        forward_outputs = self.forward_one_epoch(batch, batch_idx)
        train_loss = forward_outputs['loss']
        #print(f"train loss: {train_loss}")
        preds, labels = forward_outputs['preds'], forward_outputs['labels']
        self.accuracy(preds, labels)
        #b_input_ids = forward_outputs['input_ids']
        # Tensorboard logging for model graph and loss
        #self.logger.experiment.add_graph(self.model, input_to_model=b_input_ids, verbose=False, use_strict_trace=True)
        #self.logger.experiment.add_scalars('loss', {'train_loss': train_loss}, self.global_stepself.log("train_loss", train_loss, on_epoch=False, on_step=True, prog_bar=True)
        #`self.log("train_acc", self.accuracy, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_loss", train_loss, on_epoch=False, on_step=True, prog_bar=True)
        return {'loss': train_loss}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # update student/teacher with EMA after each batch
        #tb_size = self.hparams.batch_size * max(1, self.trainer.num_devices)
        it = self.global_step - (len(self.ds_val_ori) // self.hparams.batch_size) * self.current_epoch
        #print(it)
        self._EMA_update(it)
        #print(self.total_train_steps)
        
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        #self.evaluate(batch, "test")
        self._test(batch, "test")
        
    def configure_optimizers(self):
        # set no decay for bias and normalziation weights
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # define optimizer / scheduler
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
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
        train_path_ori = "../traindata/" + self.hparams.dataset_name_ori + "_train.csv"
        train_path_str_adv = "../traindata/" + self.hparams.dataset_name_str_adv + "_train.csv"
        train_path_weak_aug = "../traindata/" + self.hparams.dataset_name_weak_aug + "_train.csv"
        # read/generate three ways dataset
        df_train_ori = pd.read_csv(train_path_ori)
        df_train_weak_aug = pd.read_csv(train_path_weak_aug)
        df_train_str_adv = pd.read_csv(train_path_str_adv)
       
        print("Trainset Loading ...") 
        self.ds_train_ori = SequenceDataset(df_train_ori, self.hparams.dataset_name_ori, self.tokenizer,
                                            max_seq_length=self.hparams.max_seq_length)
        self.ds_train_weak_aug = SequenceDataset(df_train_weak_aug, self.hparams.dataset_name_str_adv, self.tokenizer,
                                                 max_seq_length=self.hparams.max_seq_length)
        self.ds_train_str_adv = SequenceDataset(df_train_str_adv, self.hparams.dataset_name_weak_aug, self.tokenizer,
                                                max_seq_length=self.hparams.max_seq_length)
        
        # val dataset assign
        val_path_ori = "../traindata/" + self.hparams.dataset_name_ori + "_val.csv"
        val_path_str_adv = "../traindata/" + self.hparams.dataset_name_str_adv + "_val.csv"
        val_path_weak_aug = "../traindata/" + self.hparams.dataset_name_weak_aug + "_val.csv"
        
        df_val_ori = pd.read_csv(val_path_ori)
        df_val_weak_aug = pd.read_csv(val_path_weak_aug)
        df_val_str_adv = pd.read_csv(val_path_str_adv)
        
        print("Valset Loading ...")
        self.ds_val_ori = SequenceDataset(df_val_ori, self.hparams.dataset_name_ori, self.tokenizer, 
                                          max_seq_length=self.hparams.max_seq_length)
        #self.ds_val_weak_aug = SequenceDataset(df_val_weak_aug, self.hparams.dataset_name_str_adv, self.tokenizer, 
        #                                       max_seq_length=self.hparams.max_seq_length)
        #self.ds_val_str_adv = SequenceDataset(df_val_str_adv, self.hparams.dataset_name_weak_aug, self.tokenizer, 
        #                                      max_seq_length=self.hparams.max_seq_length)
        # Calculate total steps
        tb_size = self.hparams.batch_size * max(1, self.trainer.num_devices)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(df_train_ori) // tb_size) // ab_size
        print(f"total step: {self.total_steps}")
        
      if stage == "test" or stage is None:
        # test dataset assign
        test_path_ori = "../traindata/" + self.hparams.testset_name_ori + "_test.csv"
        test_path_ssmba = "../traindata/" + self.hparams.testset_name_ssmba + "_test.csv"
        test_path_eda = "../traindata/" + self.hparams.testset_name_eda + "_test.csv"
        test_path_tf = "../traindata/" + self.hparams.testset_name_tf + "_test.csv"        

        df_test_ori = pd.read_csv(test_path_ori)
        df_test_ssmba = pd.read_csv(test_path_ssmba)
        df_test_eda = pd.read_csv(test_path_eda)
        df_test_tf = pd.read_csv(test_path_tf)
         
        print("Testset Loading ...")
        self.ds_test_ori = SequenceDataset(df_test_ori, self.hparams.testset_name_ori, self.tokenizer, max_seq_length=self.hparams.max_seq_length)
        self.ds_test_ssmba = SequenceDataset(df_test_ssmba, self.hparams.testset_name_ssmba, self.tokenizer, max_seq_length=self.hparams.max_seq_length)
        self.ds_test_eda = SequenceDataset(df_test_eda, self.hparams.testset_name_eda, self.tokenizer, max_seq_length=self.hparams.max_seq_length)
        self.ds_test_tf = SequenceDataset(df_test_tf, self.hparams.testset_name_tf, self.tokenizer, max_seq_length=self.hparams.max_seq_length)

    def train_dataloader(self):
        self.ds_train_ori = DataLoader(self.ds_train_ori, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        dl_train_weak_aug = DataLoader(self.ds_train_weak_aug, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        dl_train_str_adv = DataLoader(self.ds_train_str_adv, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
       
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = cosine_scheduler(self.hparams.momentum_teacher, 1, self.hparams.max_epochs, 
                                                len(self.ds_train_ori))
       
        loaders = {"ori": self.ds_train_ori, "weak_aug": dl_train_weak_aug, "str_adv": dl_train_str_adv} 
        combined_loader = CombinedLoader(loaders, mode='min_size')
        return combined_loader
    
    def val_dataloader(self): 
        return DataLoader(self.ds_val_ori, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        dl_test_ori = DataLoader(self.ds_test_ori, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        dl_test_ssmba = DataLoader(self.ds_test_ssmba, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        dl_test_eda = DataLoader(self.ds_test_eda, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        dl_test_tf = DataLoader(self.ds_test_tf, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        
        loaders = {"ori": dl_test_ori, "ssmba": dl_test_ssmba, "eda": dl_test_eda, "tf": dl_test_tf}
        combined_loader = CombinedLoader(loaders, mode='min_size')
        return combined_loader   
 
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        # config parameters
        parser.add_argument("--accumulate_grad_batches", type=int, default=1)
        parser.add_argument("--model_name", type=str, default="bert-base-uncased")
        parser.add_argument("--dataset_name_ori", type=str, default="agnews")
        parser.add_argument("--dataset_name_str_adv", type=str, default="agnews_ssmba")
        parser.add_argument("--dataset_name_weak_aug", type=str, default="agnews_eda")
        parser.add_argument("--testset_name_ori", type=str, default="agnews")
        parser.add_argument("--testset_name_ssmba", type=str, default="agnews_ssmba")
        parser.add_argument("--testset_name_eda", type=str, default="agnews_eda")
        parser.add_argument("--testset_name_tf", type=str, default="agnews_bert_tf")
        parser.add_argument("--num_workers", type=int, default=10)
        parser.add_argument("--max_epochs", type=int, default=5)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--max_seq_length", type=int, default=256)
        parser.add_argument("--mode", type=str, default="train")
        parser.add_argument("--load_path", type=str, default=None)
        # model parameters
        parser.add_argument("--lr", type=float, default=2e-5)
        parser.add_argument("--num_labels", type=int, default=4)
        parser.add_argument("--lamb", type=float, default=0.5)
        parser.add_argument("--loss_func", choices=['l1_smooth', 'KL', 'JS'], 
                            type=str, default="l1_smooth")
        parser.add_argument("--loss_beta", type=float, default=1.0)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument('--momentum_teacher', default=0.996, type=float)
        parser.add_argument("--scheduler_name", choices=['linear', 'cosine'],
                            type=str, default="cosine")
        parser.add_argument("--top_k_layers", type=int, default=3) 
        return parser

