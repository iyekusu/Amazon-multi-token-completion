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

# Create necessary directories

if not os.path.exists(f'{DATA_PATH}/data/wiki/'):
    os.makedirs(f'{DATA_PATH}/data/wiki/')
    logging.info("Created wikipedia training dataset directory")

def process_sentence(samples):
    formatted_dict = {
            "span":[],
            "span_lower":[],
            "range":[],
            "text":[],
            "masked_text":[]
            }       
    texts = samples['text']
    for i in range(len(texts)-1):    
        arg1 = texts[i] 
        arg2 = texts[i+1]
        for span in MTC_list:    
            if arg2.startswith(span):
                combined_text = f"{arg1} {arg2}"
                start = combined_text.index(span)
                end = start + len(span)
                masked_text = f"{arg1} {arg2.replace(span, '[SEP][MASK],')}"
                formatted_dict["span"].append(span)
                formatted_dict["span_lower"].append(span.lower())
                formatted_dict["range"].append(f"[{start},{end}]")
                formatted_dict["text"].append(combined_text)
                formatted_dict["masked_text"].append(masked_text)             
    return formatted_dict

# List of spans
with open("data/MTC_list.txt") as f:
    MTC_list = f.readlines()
    MTC_list = list(map(lambda x:x.strip(), MTC_list))
print(f'Multi-token list: {MTC_list}')

wiki_sents_dataset = load_from_disk("data/wiki_sents_dataset")['train']

# Process wiki_sentences into final format: span, soan_lower ......
results = wiki_sents_dataset.map(
    process_sentence, 
    batched=True, 
    batch_size=len(wiki_sents_dataset)//100, 
    num_proc=16, 
    load_from_cache_file = True,
    remove_columns=wiki_sents_dataset.column_names
    )

df = pd.DataFrame(results,columns=['span', 'span_lower', 'range', 'text', 'masked_text'])
df.drop_duplicates(inplace=True)
df.to_csv(f'{DATA_PATH}/data/wiki/train_comma.csv', index=False)
print("wikipedia csv saved")