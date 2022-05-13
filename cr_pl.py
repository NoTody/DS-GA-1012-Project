import os
import sys
import torch
import copy
import argparse
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from loss_utils import _jensen_shannon_div, _kl_div

class CRTransformer(LightningModule):
    def __init__(self, hparams):

        super().__init__()
        # Set our init args as class attributes
        self.hparams.update(vars(hparams))
        
        # Build models
        self.__build_model()

        # Define metrics
        self.accuracy_train = Accuracy()
        self.accuracy_val = Accuracy()
        self.accuracy_ori = Accuracy()
        self.accuracy_ssmba = Accuracy()
        self.accuracy_eda = Accuracy()
        self.accuracy_tf = Accuracy()
    
    #############################
    # Build Model Functions
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
        if self.hparams.use_ema:
            self.teacher = copy.deepcopy(self.student)
        else:
            self.teacher = self.student
 
        # there is no backpropagation through the teacher, so no need for gradients
        if self.hparams.use_projector:
            if self.hparams.use_ema:
                self.projector_teacher = self.Projector_v2(self.hparams.embed_dim * (self.hparams.top_k_layers - 1))
                self.projector_student = self.Projector_v2(self.hparams.embed_dim * (self.hparams.top_k_layers - 1))
            else:
                self.projector = self.Projector_v2(self.hparams.embed_dim * (self.hparams.top_k_layers - 1))

        if self.hparams.use_ema:
            for p in self.teacher.parameters():
                p.requires_grad = False
            if self.hparams.use_projector:
                for p in self.projector_teacher.parameters():
                    p.requires_grad = False
 
    def Projector_v1(self, embedding):
        mlp_spec = f"{embedding}-{self.hparams.mlp}"
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
            layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)

    def Projector_v2(self, embedding):
        mlp_spec = f"{embedding}-{self.hparams.mlp}"
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i+1]))
            layers.append(nn.BatchNorm1d(f[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def _EMA_update(self, it):
        """
        Exponential Moving Average update of the student
        """
        # momentum parameter
        m = self.momentum_schedule[it]

        # update weight for backbone model
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        
        # update weight for projector
        if self.hparams.use_projector:
            for param_q, param_k in zip(self.projector_student.parameters(), self.projector_teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    #############################
    # Training / Validation HOOKS
    #############################

    def forward(self, **inputs):
        return self.student(**inputs)

    def forward_one_epoch(self, batch, batch_idx, stage):
        input_ids, attn_mask, labels = batch['ori']['input_ids'], batch['ori']['attention_mask'], batch['ori']['labels']
        #print(f"input_ids: {input_ids.shape}")
        outputs_ori = self.student(input_ids, attn_mask)
 
        input_ids, attn_mask, labels = batch['str_adv']['input_ids'], batch['str_adv']['attention_mask'], batch['str_adv']['labels']
        outputs_str_adv = self.teacher(input_ids, attn_mask)

        input_ids, attn_mask, labels = batch['weak_aug']['input_ids'], batch['weak_aug']['attention_mask'], batch['weak_aug']['labels']
        outputs_weak_aug = self.student(input_ids, attn_mask)

        input_ids, attn_mask, labels = batch['eda']['input_ids'], batch['eda']['attention_mask'], batch['eda']['labels']
        outputs_eda = self.student(input_ids, attn_mask)

        criterion = nn.CrossEntropyLoss()
 
        if self.hparams.loss_func == 'l1_smooth' or self.hparams.loss_func == 'softmax': 
            # get hidden states
            hidden_states_weak_aug = outputs_weak_aug.hidden_states
            hidden_states_str_adv = outputs_str_adv.hidden_states
 
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
            if self.hparams.use_projector:
                y_aug, y_adv = y_aug.view(y_aug.size(0), -1), y_adv.view(y_adv.size(0), -1)
                if self.hparams.use_ema:
                    y_aug, y_adv = self.projector_student(y_aug), self.projector_teacher(y_adv)
                else:
                    y_aug, y_adv = self.projector(y_aug), self.projector(y_adv)
            # calculate scaled smooth l1 loss
            sz = y_aug.size(-1)
            loss_scale = 1 / (math.sqrt(sz))
            if self.hparams.loss_func == 'l1_smooth':
                selfsup_loss = loss_scale * F.smooth_l1_loss(y_aug.float(), y_adv.float(), reduction="none", beta=self.hparams.loss_beta).sum(dim=-1).sum()
            elif self.hparams.loss_func == 'softmax':
                kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
                selfsup_loss = kl_loss(F.log_softmax(y_adv, dim=-1), F.softmax(y_aug, dim=-1))
            else:
                raise NotImplementedError()
            # calculate supervised loss from true label
            logits_1 = outputs_ori.logits
            logits_2 = outputs_eda.logits
            sup_loss = self.hparams.lamb * (criterion(logits_1, labels) + criterion(logits_2, labels))
            # calculate final loss
            loss = selfsup_loss + sup_loss
            # get predict
            preds = torch.argmax(logits_1, dim=1)
        elif self.hparams.loss_func == 'JS':
            # get all logits
            logits_ori = outputs_ori.logits
            logits_str_adv = outputs_str_adv.logits
            logits_weak_aug = outputs_weak_aug.logits
            # calculate consistency regularization 
            if self.hparams.loss_func == 'JS':
                selfsup_loss = _jensen_shannon_div(logits_str_adv, logits_weak_aug, self.hparams.T)
                selfsup_loss *= self.hparams.con_coeff
            # calculate supervised loss from true label
            logits_1 = outputs_ori.logits
            logits_2 = outputs_eda.logits
            if logits_1.shape[0] != logits_2.shape[0]:
                print(logits_1)
                print(logits_2)
            sup_loss = self.hparams.lamb * (criterion(logits_1, labels) + criterion(logits_2, labels))
            # calculate final loss
            loss = selfsup_loss + sup_loss
            # get predict
            preds = torch.argmax(logits_1, dim=1)
        else:
            raise NotImplementedError()
        # logs
        if stage == "train":
            self.log(f"train_suploss", sup_loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log(f"train_selfsuploss", selfsup_loss, on_step=True, on_epoch=False, prog_bar=True)

        return {'loss': loss, 'preds': preds, 'labels': labels}
 
    def _calc_test_metrics(self, batch, loader_name):
        input_ids, attn_mask, labels = batch[loader_name]['input_ids'], batch[loader_name]['attention_mask'], batch[loader_name]['labels']
        outputs_ori = self.student(input_ids, attn_mask)
        logits = outputs_ori.logits
        hidden_states = outputs_ori.hidden_states
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels 

    def _test(self, batch, stage=None):
        # ori test
        loss_ori, preds_ori, labels = self._calc_test_metrics(batch, 'ori')
        self.accuracy_ori(preds_ori, labels)
        if stage:
            self.log(f"{stage}_loss_ori", loss_ori, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc_ori", self.accuracy_ori, on_step=False, on_epoch=True, prog_bar=True)
        
        # ssmba test
        loss_ssmba, preds_ssmba, labels = self._calc_test_metrics(batch, 'ssmba')
        self.accuracy_ssmba(preds_ssmba, labels)
        if stage:
            self.log(f"{stage}_loss_ssmba", loss_ssmba, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc_ssmba", self.accuracy_ssmba, on_step=False, on_epoch=True, prog_bar=True)

        # eda test
        loss_eda, preds_eda, labels = self._calc_test_metrics(batch, 'eda')
        self.accuracy_eda(preds_eda, labels)
        if stage:
            self.log(f"{stage}_loss_eda", loss_eda, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc_eda", self.accuracy_eda, on_step=False, on_epoch=True, prog_bar=True)

        # textfooler test
        loss_tf, preds_tf, labels = self._calc_test_metrics(batch, 'tf')
        self.accuracy_tf(preds_tf, labels)
        if stage:
            self.log(f"{stage}_loss_tf", loss_tf, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc_tf", self.accuracy_tf, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        forward_outputs = self.forward_one_epoch(batch, batch_idx, "train")
        train_loss = forward_outputs['loss']
        preds, labels = forward_outputs['preds'], forward_outputs['labels']
        self.accuracy_train(preds, labels)
        self.log("train_loss", train_loss, on_epoch=False, on_step=True, prog_bar=True)
        return {'loss': train_loss}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # update student/teacher with EMA after each batch
        it = self.global_step - (len(self.ds_val_ori) // self.hparams.batch_size) * self.current_epoch
        if self.hparams.use_ema:
            self._EMA_update(it)
        
    def validation_step(self, batch, batch_idx):
        forward_outputs = self.forward_one_epoch(batch, batch_idx, "val")
        val_loss = forward_outputs['loss']
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True) 
    
    def test_step(self, batch, batch_idx):
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
        # define optimizer 
        if self.hparams.use_projector:
            if self.hparams.use_ema:
                optimizer_s = AdamW([{"params": self.student.parameters()}, {"params": self.projector_student.parameters(), "lr": self.hparams.lr_projector}], lr=self.hparams.lr_backbone)
            else:
                optimizer_s = AdamW([{"params": self.student.parameters()}, {"params": self.projector.parameters(), "lr": self.hparams.lr_projector}], lr=self.hparams.lr_backbone)
        else:
            optimizer_s = AdamW(self.student.parameters(), lr=self.hparams.lr_backbone)
        
        # warmup and scheduler setup
        self.warmup_steps = 0.02 * self.total_steps
        if self.hparams.scheduler_name == "cosine":
            scheduler_s = get_cosine_schedule_with_warmup(
                optimizer_s,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.hparams.scheduler_name == "linear":
            scheduler_s = get_linear_schedule_with_warmup(
                optimizer_s,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
            )
            scheduler_p = CosineAnnealingLR(optimizer_p, eta_min=1e-5)
        return [optimizer_s], [scheduler_s]

    ####################
    # DATA RELATED HOOKS
    ####################
    def setup(self, stage=None):
      # dataset setup
      if stage == "fit" or stage is None:
        # train dataset assign
        train_path_ori = self.hparams.data_dir + self.hparams.dataset_name_ori + "_train.csv"
        train_path_str_adv = self.hparams.data_dir + self.hparams.dataset_name_str_adv + "_train.csv"
        train_path_weak_aug = self.hparams.data_dir + self.hparams.dataset_name_weak_aug + "_train.csv"
        train_path_eda = self.hparams.data_dir + self.hparams.dataset_name_eda + "_train.csv"
        # read/generate three ways dataset
        df_train_ori = pd.read_csv(train_path_ori)
        df_train_weak_aug = pd.read_csv(train_path_weak_aug)
        df_train_str_adv = pd.read_csv(train_path_str_adv)
        df_train_eda = pd.read_csv(train_path_eda)
 
        print("Trainset Loading ...") 
        self.ds_train_ori = SequenceDataset(df_train_ori, self.hparams.dataset_name_ori, self.tokenizer,
                                            max_seq_length=self.hparams.max_seq_length)
        self.ds_train_weak_aug = SequenceDataset(df_train_weak_aug, self.hparams.dataset_name_str_adv, self.tokenizer,
                                                 max_seq_length=self.hparams.max_seq_length)
        self.ds_train_str_adv = SequenceDataset(df_train_str_adv, self.hparams.dataset_name_weak_aug, self.tokenizer,
                                                max_seq_length=self.hparams.max_seq_length)
        self.ds_train_eda = SequenceDataset(df_train_eda, self.hparams.dataset_name_eda, self.tokenizer,
                                            max_seq_length=self.hparams.max_seq_length)
        # val dataset assign
        val_path_ori = self.hparams.data_dir + self.hparams.dataset_name_ori + "_val.csv"
        val_path_str_adv = self.hparams.data_dir + self.hparams.dataset_name_str_adv + "_val.csv"
        val_path_weak_aug = self.hparams.data_dir + self.hparams.dataset_name_weak_aug + "_val.csv"
        val_path_eda = self.hparams.data_dir + self.hparams.dataset_name_eda + "_val.csv"

        df_val_ori = pd.read_csv(val_path_ori)
        df_val_weak_aug = pd.read_csv(val_path_weak_aug)
        df_val_str_adv = pd.read_csv(val_path_str_adv)
        df_val_eda = pd.read_csv(val_path_eda)        

        print("Valset Loading ...")
        self.ds_val_ori = SequenceDataset(df_val_ori, self.hparams.dataset_name_ori, self.tokenizer, 
                                        max_seq_length=self.hparams.max_seq_length)
        self.ds_val_weak_aug = SequenceDataset(df_val_weak_aug, self.hparams.dataset_name_str_adv, self.tokenizer, 
                                            max_seq_length=self.hparams.max_seq_length)
        self.ds_val_str_adv = SequenceDataset(df_val_str_adv, self.hparams.dataset_name_weak_aug, self.tokenizer, 
                                            max_seq_length=self.hparams.max_seq_length)
        self.ds_val_eda = SequenceDataset(df_val_eda, self.hparams.dataset_name_eda, self.tokenizer, 
                                        max_seq_length=self.hparams.max_seq_length)
        # Calculate total steps
        tb_size = self.hparams.batch_size * max(1, self.trainer.num_devices)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(df_train_ori) // tb_size) * ab_size
        print(f"total steps: {self.total_steps}")
        
      if stage == "test" or stage is None:
        # test dataset assign
        test_path_ori = self.hparams.data_dir + self.hparams.testset_name_ori + "_test.csv"
        test_path_ssmba = self.hparams.data_dir + self.hparams.testset_name_ssmba + "_test.csv"
        test_path_eda = self.hparams.data_dir + self.hparams.testset_name_eda + "_test.csv"
        test_path_tf = self.hparams.data_dir + self.hparams.testset_name_tf + "_test.csv"        

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
        self.ds_train_ori = DataLoader(self.ds_train_ori, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=True)
        dl_train_weak_aug = DataLoader(self.ds_train_weak_aug, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=True)
        dl_train_str_adv = DataLoader(self.ds_train_str_adv, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=True)
        dl_train_eda = DataLoader(self.ds_train_eda, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=True)
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = cosine_scheduler(self.hparams.momentum_teacher, 1, self.hparams.max_epochs, 
                                                len(self.ds_train_ori))
       
        loaders = {"ori": self.ds_train_ori, "str_adv": dl_train_str_adv, "weak_aug": dl_train_weak_aug, "eda": dl_train_eda} 
        combined_loader = CombinedLoader(loaders, mode='min_size')
        return combined_loader
    
    def val_dataloader(self): 
        self.ds_val_ori = DataLoader(self.ds_val_ori, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        dl_val_weak_aug = DataLoader(self.ds_val_weak_aug, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        dl_val_str_adv = DataLoader(self.ds_val_str_adv, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        dl_val_eda = DataLoader(self.ds_val_eda, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

        loaders = {"ori": self.ds_val_ori, "str_adv": dl_val_str_adv, "weak_aug": dl_val_weak_aug , "eda": dl_val_eda}
        combined_loader = CombinedLoader(loaders, mode='min_size')
        return combined_loader

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
        parser.add_argument("--data_dir", type=str, default="../traindata/")
        parser.add_argument("--model_name", type=str, default="bert-base-uncased")
        parser.add_argument("--dataset_name_ori", type=str, default="agnews")
        parser.add_argument("--dataset_name_str_adv", type=str, default="agnews_ssmba")
        parser.add_argument("--dataset_name_weak_aug", type=str, default="agnews_eda")
        parser.add_argument("--dataset_name_eda", type=str, default="agnews_eda")
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
        parser.add_argument("--lr_backbone", type=float, default=4e-5)
        parser.add_argument("--lr_projector", type=float, default=1e-3)
        parser.add_argument("--num_labels", type=int, default=4)
        parser.add_argument("--lamb", type=float, default=1.0)
        parser.add_argument("--mlp", type=str, default="4096-4096-4096")
        parser.add_argument("--loss_func", choices=['l1_smooth', 'KL', 'softmax'],
                            type=str, default="l1_smooth")
        parser.add_argument("--loss_beta", type=float, default=1.0)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument('--momentum_teacher', default=0.996, type=float)
        parser.add_argument("--scheduler_name", choices=['linear', 'cosine'],
                            type=str, default="cosine")
        parser.add_argument("--top_k_layers", type=int, default=4)
        parser.add_argument("--T", type=float, default=0.5) 
        parser.add_argument("--con_coeff", type=float, default=1.0)
        parser.add_argument("--embed_dim", type=int, default=768)
        parser.add_argument("--use_ema", action='store_true')
        parser.add_argument("--use_projector", action='store_true')
        return parser

