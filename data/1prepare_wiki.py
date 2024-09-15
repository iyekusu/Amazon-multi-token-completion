from datasets import load_dataset, load_from_disk
import pandas as pd
import re
import os
import csv
import sys
import logging
from collections import Counter

from multiprocessing import Pool, cpu_count

import spacy
import json
from random import choice
from spacy.language import Language
from glob import glob

sys.path.append(os.path.abspath('.'))
from configuration import DATA_PATH

SAMPLE = int(1e6)

num_cpus = cpu_count()
print(num_cpus)

# Verify DATA_PATH and create necessary directories
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    logging.info(f"Created DATA_PATH directory at {DATA_PATH}")

if not os.path.exists(f'{DATA_PATH}/data/wiki_sents/'):
    os.makedirs(f'{DATA_PATH}/data/wiki_sents/')
    logging.info("Created wiki_sents directory")

if not os.path.exists(f'{DATA_PATH}/data/wiki/'):
    os.makedirs(f'{DATA_PATH}/data/wiki/')
    logging.info("Created wikipedia training dataset directory")

@Language.component('set_custom_boundaries')
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if '\n' in token.text:
            doc[token.i + 1].is_sent_start = True
    return doc

def get_filtered_sents(nlp, s):
    doc = nlp(s)
    sents = [s.text.strip() for s in doc.sents if 5 <= len(s) <= 100 and not s.text.startswith('Category:') and 'VERB' in [t.pos_ for t in s]]
    if sents and 'may refer to:' in sents[0]:
        sents = []
    return sents

def samples_to_sentences(samples, inds):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('set_custom_boundaries', first=True)
    with open(f'{DATA_PATH}/data/wiki_sents/wiki_sentences{inds[0]}_{inds[-1]}.jsonl', 'w') as fp:
        for i, s in zip(inds, samples['text']):
            sents = get_filtered_sents(nlp, s)
            for sent in sents:
                fp.write(json.dumps({'ind': i, 'text': sent}))
                fp.write('\n')

wiki_dataset = load_from_disk("data/wiki_dataset")['train']

# Process wikipedia articles into sentences
wiki_dataset.map(
    samples_to_sentences, 
    with_indices=True, 
    batched=True, 
    batch_size=len(wiki_dataset) // 100, 
    num_proc=16, 
    load_from_cache_file=True,
    remove_columns=wiki_dataset.column_names
    )
dataset = load_dataset('json', data_files={'train': glob(f'data/wiki_sents/wiki_sentences*.jsonl')})
dataset.save_to_disk(f'data/wiki_sents_dataset/')
print(dataset.column_names)