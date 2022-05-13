import os
import sys
import torch
import logging
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

class LitTransformer(LightningModule):
    def __init__(self, args):

        super().__init__()
   
        # Set our init args as class attributes
        self.num_devices = args.num_devices
        self.accumulate_grad_batches = args.accumulate_grad_batches
        self.dataset_name_ori = args.dataset_name_ori
        self.dataset_name_eda = args.dataset_name_eda
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_seq_length = args.max_seq_length
        self.model_name = args.model_name
        
        self.testset_name_ori = args.testset_name_ori
        self.testset_name_ssmba = args.testset_name_ssmba
        self.testset_name_eda = args.testset_name_eda
        self.testset_name_tf = args.testset_name_tf
        
        # Dataset specific attributes
        self.num_labels = args.num_labels
        self.num_workers = args.num_workers

        # Define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                       use_fast=True)

        # Define config
        self.config = AutoConfig.from_pretrained(self.model_name, 
                                                 num_labels=self.num_labels,
                                                 output_hidden_states=True)

        # Define hyperparameters
        self.weight_decay = args.weight_decay
        self.scheduler_name = args.scheduler_name

        # Define PyTorch model
        self.model = model_init(self.model_name, self.config)

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

    def forward(self, **inputs):
        return self.model(**inputs)

    def forward_one_epoch(self, batch, batch_idx):
        # ori loss
        input_ids, attn_mask, labels = batch['ori']['input_ids'], batch['ori']['attention_mask'], batch['ori']['labels']
        outputs_ori = self.model(input_ids, attn_mask)
        logits = outputs_ori.logits
        hidden_states = outputs_ori.hidden_states
        criterion = nn.CrossEntropyLoss()
        loss_ori = criterion(logits, labels)
        # eda loss
        input_ids, attn_mask, labels = batch['eda']['input_ids'], batch['eda']['attention_mask'], batch['eda']['labels']
        outputs_eda = self.model(input_ids, attn_mask)
        logits = outputs_eda.logits
        hidden_states = outputs_eda.hidden_states
        loss_eda = criterion(logits, labels)
        # total loss
        loss = loss_ori + loss_eda
        preds = torch.argmax(logits, dim=1)
        return {'loss': loss, 'preds': preds, 'input_ids': input_ids, 
                'labels': labels, 'logits': logits, 'hidden_states': hidden_states}

    def _test(self, batch, stage=None):
        input_ids, attn_mask, labels = batch['ori']['input_ids'], batch['ori']['attention_mask'], batch['ori']['labels']
        outputs_ori = self.model(input_ids, attn_mask)
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
        outputs_ssmba = self.model(input_ids, attn_mask)
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
        outputs_eda = self.model(input_ids, attn_mask)
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
        outputs_tf = self.model(input_ids, attn_mask)
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
        # Tensorboard logging for model graph and loss
        self.log("train_loss", train_loss, on_epoch=False, on_step=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        forward_outputs = self.forward_one_epoch(batch, batch_idx)
        val_loss = forward_outputs['loss']
        preds = forward_outputs['preds']
        labels = forward_outputs['labels']
        self.accuracy(preds, labels)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        self._test(batch, stage="test")

    def configure_optimizers(self):
        # set no decay for bias and normalziation weights
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # define optimizer / scheduler
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        self.warmup_steps = 0.02 * self.total_steps
        if self.scheduler_name == "cosine":
          scheduler = get_cosine_schedule_with_warmup(
              optimizer,
              num_warmup_steps=self.warmup_steps,
              num_training_steps=self.total_steps,
          )
        elif self.scheduler_name == "linear":
          scheduler = get_linear_schedule_with_warmup(
              optimizer,
              num_warmup_steps=self.warmup_steps,
              num_training_steps=self.total_steps,
          )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    ####################
    # DATA RELATED HOOKS
    ####################
    def setup(self, stage=None):
      # dataset setup
      if stage == "fit" or stage is None:
        # train dataset assign
        train_path_ori = "../traindata/" + self.dataset_name_ori + "_train.csv"
        df_train_ori = pd.read_csv(train_path_ori)
        logging.info("Preparing ori training data...")
        self.ds_train_ori = SequenceDataset(df_train_ori, self.dataset_name_ori, self.tokenizer, max_seq_length=self.max_seq_length)

        train_path_eda = "../traindata/" + self.dataset_name_eda + "_train.csv"
        df_train_eda = pd.read_csv(train_path_eda)
        logging.info("Preparing eda training data...")
        self.ds_train_eda = SequenceDataset(df_train_eda, self.dataset_name_eda, self.tokenizer, max_seq_length=self.max_seq_length)
 
        # val dataset assign
        val_path_ori = "../traindata/" + self.dataset_name_ori + "_val.csv"
        df_val_ori = pd.read_csv(val_path_ori)
        logging.info("Preparing ori validation data...")
        self.ds_val_ori = SequenceDataset(df_val_ori, self.dataset_name_ori, self.tokenizer, max_seq_length=self.max_seq_length)

        val_path_eda = "../traindata/" + self.dataset_name_eda + "_val.csv"
        df_val_eda = pd.read_csv(val_path_eda)
        logging.info("Preparing eda validation data...")
        self.ds_val_eda = SequenceDataset(df_val_eda, self.dataset_name_eda, self.tokenizer, max_seq_length=self.max_seq_length)
 
        # Calculate total steps
        tb_size = self.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(df_train_ori) // tb_size) * ab_size
        print(f"total step: {self.total_steps}")

      if stage == "test" or stage is None:
        # test dataset assign
        test_path_ori = "../traindata/" + self.testset_name_ori + "_test.csv"
        test_path_ssmba = "../traindata/" + self.testset_name_ssmba + "_test.csv"
        test_path_eda = "../traindata/" + self.testset_name_eda + "_test.csv"
        test_path_tf = "../traindata/" + self.testset_name_tf + "_test.csv"        

        df_test_ori = pd.read_csv(test_path_ori)
        df_test_ssmba = pd.read_csv(test_path_ssmba)
        df_test_eda = pd.read_csv(test_path_eda)
        df_test_tf = pd.read_csv(test_path_tf)

        print("Testset Loading ...")
        self.ds_test_ori = SequenceDataset(df_test_ori, self.testset_name_ori, self.tokenizer, max_seq_length=self.max_seq_length)
        self.ds_test_ssmba = SequenceDataset(df_test_ssmba, self.testset_name_ssmba, self.tokenizer, max_seq_length=self.max_seq_length)
        self.ds_test_eda = SequenceDataset(df_test_eda, self.testset_name_eda, self.tokenizer, max_seq_length=self.max_seq_length)
        self.ds_test_tf = SequenceDataset(df_test_tf, self.testset_name_tf, self.tokenizer, max_seq_length=self.max_seq_length)

    def train_dataloader(self):
        dl_train_ori = DataLoader(self.ds_train_ori, batch_size=self.batch_size, num_workers=self.num_workers)
        dl_train_eda = DataLoader(self.ds_train_eda, batch_size=self.batch_size, num_workers=self.num_workers)
        loaders = {"ori": dl_train_ori, "eda": dl_train_eda}
        combined_loader = CombinedLoader(loaders, mode='min_size')
        return combined_loader

    def val_dataloader(self):
        dl_val_ori = DataLoader(self.ds_val_ori, batch_size=self.batch_size, num_workers=self.num_workers)
        dl_val_eda = DataLoader(self.ds_val_eda, batch_size=self.batch_size, num_workers=self.num_workers)
        loaders = {"ori": dl_val_ori, "eda": dl_val_eda}
        combined_loader = CombinedLoader(loaders, mode='min_size')
        return combined_loader

    def test_dataloader(self):
        dl_test_ori = DataLoader(self.ds_test_ori, batch_size=self.batch_size, num_workers=self.num_workers)
        dl_test_ssmba = DataLoader(self.ds_test_ssmba, batch_size=self.batch_size, num_workers=self.num_workers)
        dl_test_eda = DataLoader(self.ds_test_eda, batch_size=self.batch_size, num_workers=self.num_workers)
        dl_test_tf = DataLoader(self.ds_test_tf, batch_size=self.batch_size, num_workers=self.num_workers)
 
        loaders = {"ori": dl_test_ori, "ssmba": dl_test_ssmba, "eda": dl_test_eda, "tf": dl_test_tf}
        combined_loader = CombinedLoader(loaders, mode='min_size')
        return combined_loader
 
