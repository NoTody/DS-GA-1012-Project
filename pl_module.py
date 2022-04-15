import os
import sys
import torch
import logging
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

class LitTransformer(LightningModule):
    def __init__(self, args):

        super().__init__()
   
        # Set our init args as class attributes
        self.num_devices = args.num_devices
        self.accumulate_grad_batches = args.accumulate_grad_batches
        self.dataset_name = args.dataset_name
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_seq_length = args.max_seq_length
        self.model_name = args.model_name

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
        self.f1 = F1Score()

    #############################
    # Training / Validation HOOKS
    #############################

    def forward(self, **inputs):
        return self.model(**inputs)

    def forward_one_epoch(self, batch, batch_idx):
        b_input_ids, b_attn_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        outputs = self.model(b_input_ids, b_attn_mask)
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, b_labels)
        preds = torch.argmax(logits, dim=1)
        return {'loss': loss, 'preds': preds, 'input_ids': b_input_ids, 
                'labels': b_labels, 'logits': logits, 'hidden_states': hidden_states}

    def training_step(self, batch, batch_idx):
        forward_outputs = self.forward_one_epoch(batch, batch_idx)
        train_loss = forward_outputs['loss']
        b_input_ids = forward_outputs['input_ids']
        # Tensorboard logging for model graph and loss
        #self.logger.experiment.add_graph(self.model, input_to_model=b_input_ids, verbose=False, use_strict_trace=True)
        #self.logger.experiment.add_scalars('loss', {'train_loss': train_loss}, self.global_step)
        self.log("train_loss", train_loss, on_epoch=False, on_step=True, prog_bar=True)
        return train_loss

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
        self.warmup_steps = 0.06 * self.total_steps
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
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    ####################
    # DATA RELATED HOOKS
    ####################
    def setup(self, stage=None):
      # dataset setup
      if stage == "fit" or stage is None:
        # train dataset assign
        train_path = "./datasets/" + self.dataset_name + "_train.csv"
        df_train = pd.read_csv(train_path)
        logging.info("Preparing training data...")
        self.ds_train = SequenceDataset(df_train, self.dataset_name, self.tokenizer, max_seq_length=self.max_seq_length)
        # val dataset assign
        val_path = "./datasets/" + self.dataset_name + "_val.csv"
        df_val = pd.read_csv(val_path)
        logging.info("Preparing validation data...")
        self.ds_val = SequenceDataset(df_val, self.dataset_name, self.tokenizer, max_seq_length=self.max_seq_length)
        # Calculate total steps
        tb_size = self.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(df_train) // tb_size) // ab_size
        print(f"total step: {self.total_steps}")
      if stage == "test" or stage is None:
        # test dataset assign
        test_path = "./datasets/" + self.dataset_name + "_test.csv"
        df_test = pd.read_csv(test_path)
        logging.info("Preparing test data...")
        self.ds_test = SequenceDataset(df_test, self.dataset_name, self.tokenizer, max_seq_length=self.max_seq_length)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers)
