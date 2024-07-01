import argparse
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import csv
import json

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

from sense_mapping import sense_dict


def create_mapping(dataset, model_name, dataset_path, model_path, dataset_suffix):
    tok: BertTokenizerFast = AutoTokenizer.from_pretrained(model_name)
    seen_dataset = datasets.DatasetDict({k: v for k, v in dataset.items() if k in ['train', 'dev', 'test'] and k in dataset})
    with open('uni_MTC.txt', 'r') as f:
        vocab = set(f.read().splitlines())
    print(f'Vocab before mapping: {len(tok.vocab)}')

    mapping = tok.vocab
    cur_id = len(mapping)
    for v in tqdm(vocab):
        if v not in mapping:
            mapping[v] = cur_id
            cur_id += 1

    seen_dataset = seen_dataset.map(lambda samples: {'labels': [mapping[v] for v in samples['span']]}, batched=True, num_proc=16)
    seen_dataset.save_to_disk(dataset_path)

    # Load the model and adjust embeddings
    model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_name)
    print(f'Extended vocab: {len(mapping)}')
    model.resize_token_embeddings(len(mapping))
    model.save_pretrained(model_path)

    pd.Series(mapping).to_csv(f'matrix_plugin_v0/ext_dec_map_{model_name.replace("/", "_")}{dataset_suffix}.csv', index_label='phrase', header=['id'])
    return model, seen_dataset, mapping

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
        , logger=TensorBoardLogger(save_dir=os.getcwd(), version=config['version'], name='matrix_plugin_v0_results')
        , max_epochs=config['epochs']
        , val_check_interval=config['val_check_interval']
        , gradient_clip_val=1
    )

    model = MatrixDecoder(config, transformer_matrix)
    dataset.set_format(type='pt', columns=['input_features', 'labels'])
    trainer.fit(model,
                DataLoader(dataset['train'], config['batch_size'], num_workers=16, shuffle=True),
                DataLoader(dataset['dev'].select(range(config['dev_size'])), 32, num_workers=16))

    return trainer.checkpoint_callback.best_model_path


class Logger:
    def __init__(self, logfile):
        print("output to:", logfile)
        self.logfile = logfile
        self.fp = open(logfile, 'w')

    def print(self, output):
        print(output)
        self.fp.write(str(output) + '\n')

    def close(self):
        print("output printed to:", self.logfile)
        self.fp.close()

@torch.no_grad()
def test(matrix, dataset: datasets.DatasetDict, mapping: Dict, tokenizer: BertTokenizerFast, ckpt_path, sense_dict,log=False):
    test_model = MatrixDecoder.load_from_checkpoint(ckpt_path, model=matrix).cuda().eval()
    test_dataset = seen_dataset['test']
    rev_map = {v: k for k, v in mapping.items()}
    acc_at_to_check = [1, 2, 3, 5, 10, 20, 50]
    found, total = defaultdict(int), 0
    PRINT_EVERY = 20
    
    print(f'sense_dict: {sense_dict}')
    print(f'sense_dict: {type(sense_dict)}')

    logger = Logger(Path(ckpt_path).parent.parent / 'test.log')
    if log:
        top5_fp = open(Path(ckpt_path).parent.parent / 'top5_res.csv', 'w')
        writer = csv.DictWriter(top5_fp, fieldnames=['top5_match', 'masked', 'span', 'top5_results'])
        writer.writeheader()

    BATCH_SZ = 128
    loader = DataLoader(test_dataset, batch_size=BATCH_SZ, shuffle=False) 
    for irow, row in enumerate(tqdm(loader)):
        # if not isinstance(sense_dict, dict):
        #     print(sense_dict)

        # metadata
        filenames = row['filename']
        gold_senses = row['sense1'] 
        texts = row['text']
        masked_texts = row['masked_text']
        # predictions
        results = torch.topk(test_model.forward(torch.stack(row['input_features']).type(torch.FloatTensor).T.cuda()), k=max(acc_at_to_check), dim=-1)
        text_res = [[rev_map[int(i)] for i in res] for res in results.indices]
        total += len(row['span'])
        spans = tokenizer.batch_decode(tokenizer(row['span'])['input_ids'], skip_special_tokens=True)

        # Process each sample in the batch
        for i, span in enumerate(spans):
            correct = False
            gold_sense = gold_senses[i]
            predicted_connectives = [rev_map[idx] for idx in results.indices[i][:max(acc_at_to_check)]]

            for predicted in predicted_connectives:
                if gold_sense in sense_dict.get(predicted.lower(), []):
                    correct = True
                    break  # Stop checking once a correct prediction is found

            for acc_at in acc_at_to_check:
                if correct and len(predicted_connectives) >= acc_at:
                    found[acc_at] += 1

            if log:
                writer.writerow({
                    'top5_match': 'V' if correct else 'X',
                    'masked': row['masked_text'][i],
                    'span': span,
                    'top5_results': [rev_map.get(idx, '') for idx in results.indices[i][:5]]
                })

        if irow % PRINT_EVERY == 0 and irow > 0:
            logger.print(" ".join(f"accuracy at {acc_at}: {found[acc_at] / total:.2%}" for acc_at in acc_at_to_check))

    # Final log output after processing all batches
    for acc_at in acc_at_to_check:
        logger.print(f"Final accuracy at {acc_at}: {found[acc_at] / total:.2%}")

    if log:
        top5_fp.close()
        # Process each sample in the batch
    #     for i, span in enumerate(spans):
    #         gold_sense = row['sense1'][i]  # Accessing gold sense for each sample
    #         predicted_connectives = text_res[i][:max(acc_at_to_check)]
    #         correct = any(gold_sense in sense_dict.get(predicted.lower(), []) for predicted in predicted_connectives)
    #         for acc_at in acc_at_to_check:
    #                 if any(gold_sense in sense_dict.get(predicted.lower(), []) for predicted in predicted_connectives[:acc_at]):
    #                     found[acc_at] += 1
    #         # Logging if enabled
    #         if log:
    #             writer.writerow({
    #                 'top5_match': 'V' if correct else 'X',
    #                 'masked': row['masked_text'][i],
    #                 'span': span,
    #                 'top5_results': text_res[i][:5]
    #             })

    #     if irow % PRINT_EVERY == 0 and irow > 0:
    #         logger.print(" ".join(f"accuracy at {acc_at}: {found[acc_at] / total:.2%}" for acc_at in acc_at_to_check))

    # # Final log output after processing all batches
    # for acc_at in acc_at_to_check:
    #     logger.print(f"Final accuracy at {acc_at}: {found[acc_at] / total:.2%}")

    # if log:
    #     top5_fp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-cased')
    parser.add_argument('--version', type=str, default='version_0')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--dataset_name', type=str, default='wiki_v0')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val_check_interval', type=float, default=0.25)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--dev_size', type=int, default=20_000)
    parser.add_argument('--epochs', type=int, default=1)
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

    model_path = f'matrix_plugin_v0/ext_dec_{model_name}{dataset_suffix}'
    dataset_path = f'data/matrix_plugin_v0/seen_dataset_{model_name}{dataset_suffix}/'

    if args.force or not os.path.exists(model_path) or not os.path.exists(dataset_path):
        if args.input_path is not None:
            input_path = args.input_path
        elif 'roberta' in model_name:
            input_path = f'{HOME_DIR}/MultiTokenCompletionData/input_data_{model_name}{dataset_suffix}'
        elif 'spanbert' in model_name:
            input_path = f'input_data_{model_name}'
        else:
            input_path = f'{HOME_DIR}/MultiTokenCompletionData/input_data_cased'

        dataset = datasets.load_from_disk(input_path)

        # Save dataset back if it was modified
        test_split_path = os.path.join(input_path, 'test')
        if 'test' in dataset and os.path.exists(test_split_path):
            dataset['test'] = datasets.load_from_disk(test_split_path)
        print('increasing vocab')
        model, seen_dataset, mapping = create_mapping(dataset, model_name, dataset_path, model_path, dataset_suffix)
        seen_dataset.save_to_disk(dataset_path)
    else:
        print('loading vocab')
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        seen_dataset = datasets.load_from_disk(dataset_path)
        mapping = pd.read_csv(f'matrix_plugin_v0/ext_dec_map_{model_name.replace("/", "_")}{dataset_suffix}.csv', na_filter=False).set_index('phrase')[
            'id'].to_dict()
        model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_path)

    if 'roberta' in model_name:
        # RobertaForMaskedLM
        matrix = model.lm_head
    else:
        # BertForMaskedLM
        matrix = model.cls

    tok = AutoTokenizer.from_pretrained(model_name)

    if args.test:
        matrix.eval()
        ckpt = args.ckpt
        if ckpt is None:
            from glob import glob
            ckpt = sorted(glob(f'matrix_plugin_v0_results/{args.version}/checkpoints/*.ckpt'))[-1]
        print(ckpt)
        print(type(sense_dict))
        print(sense_dict)
        test(matrix, seen_dataset, mapping, tok, ckpt, args.log, sense_dict)
    else:
        # detach output embedding matrix so it will train
        model.get_output_embeddings().weight = torch.nn.Parameter(model.get_output_embeddings().weight.clone())
        best_ckpt = train(config, matrix, seen_dataset)
        matrix.eval()
        test(matrix, seen_dataset, mapping, tok, best_ckpt, args.log, sense_dict)
