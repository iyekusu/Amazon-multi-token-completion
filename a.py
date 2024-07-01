import argparse
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import csv

from dataclasses import dataclass, field, asdict
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from torch import no_grad
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, BertTokenizerFast, TrainingArguments, BertTokenizer, BertForMaskedLM, \
    HfArgumentParser, RobertaForMaskedLM, BertForSequenceClassification
import datasets
import os
from tqdm import tqdm
import pandas as pd
import torch
import pytorch_lightning as pl
import torch
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from typing import Dict

class MatrixDecoder(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()

        self.save_hyperparameters(config)
        self.config = config
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_features):
        return self.model(input_features)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['lr'])

    def step(self, batch):
        y = self.forward(batch['input_features'])
        loss = self.criterion(y, batch['labels'])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss


def train(config: Dict, transformer_matrix: BertOnlyMLMHead, dataset: datasets.DatasetDict):
    trainer = pl.Trainer(
        devices=config['num_gpus']
        , accelerator='gpu'
        , logger=TensorBoardLogger(save_dir=os.getcwd(), version=config['version'], name='matrix_plugin_results')
        , max_epochs=1
        , gradient_clip_val=1
    )

    model = MatrixDecoder(config, transformer_matrix)
    dataset.set_format(type='pt', columns=['input_features', 'labels'])
    trainer.fit(model,
                DataLoader(dataset['train'], config['batch_size'], num_workers=16, shuffle=True),
                DataLoader(dataset['dev'].select(range(config['dev_size'])), 32, num_workers=16)
                )

    return trainer.checkpoint_callbacks[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-cased')
    parser.add_argument('--version', type=str, default='version_1')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--dataset_name', type=str, default='wiki_pub')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val_check_interval', type=float, default=0.25)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--dev_size', type=int, default=20_000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    model_name = args.model
    args.log = not args.no_log
    config = vars(args)

    dataset_name = args.dataset_name
    dataset_suffix = '' if dataset_name == 'wiki' else f'_{dataset_name}'
    model_path = f'matrix_plugin/ext_dec_{model_name}{dataset_suffix}'
    dataset_path = f'data/matrix_plugin/seen_dataset_{model_name}{dataset_suffix}_copy/'
    tok = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    seen_dataset = datasets.load_from_disk(dataset_path)
    mapping = pd.read_csv(f'matrix_plugin/ext_dec_map_{model_name.replace("/", "_")}{dataset_suffix}.csv', na_filter=False).set_index('phrase')[
        'id'].to_dict()
    model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_path)
    # detach output embedding matrix so it will train
    model.get_output_embeddings().weight = torch.nn.Parameter(model.get_output_embeddings().weight.clone())
    matrix = model.cls
    best_ckpt = train(config, matrix, seen_dataset)
    if best_ckpt is None:
        print("Training did not produce any checkpoints.")
    else:
        print("Best checkpoint:", best_ckpt)
    matrix.eval()
