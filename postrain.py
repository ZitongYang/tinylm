import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Dict
from functools import partial
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-Base",
        use_fast=True)
    return tokenizer

def _write_memmap(dset: Dataset, filename: str):
    dtype = np.int32
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    print(f'Writing to {filename} with length {arr_len}')
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = min(1024, len(dset))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
        arr.flush()

def process(example: Dict, tokenizer: AutoTokenizer, max_length: int = 512)->Dict:
    """
    Tokenize the text and return the tokenized text
    """
    messages = example['conversations']
    result = []
    for message in messages:
        if message['from'] == 'human':
            message['from'] = 'user'
        elif message['from'] == 'gpt':
            message['from'] = 'assistant'
        result.append(dict(role=message['from'], content=message['value']))

    ids_without_truncation = tokenizer.apply_chat_template(result, add_generation_prompt=True)
    if len(ids_without_truncation) <= max_length:
        ids = tokenizer.apply_chat_template(result, padding='max_length', max_length=max_length)
        return dict(ids=ids, len=len(ids))
    else:
        return dict(ids=[], len=0)

def tokenize_postrain():
    filename = f'postrain.bin'
    if os.path.exists(filename):
        print(f'{filename} already exists')
    else:
        tokenizer = get_tokenizer()
        process_map = partial(process, tokenizer=tokenizer)
        # loading dataset
        dataset = load_dataset("teknium/OpenHermes-2.5",
                            trust_remote_code=True)
        # core tokenization operation happening
        tokenized_train = dataset['train'].map(
            process_map,
            remove_columns=dataset['train'][0].keys(),
            desc='Tokenizing training split',
            num_proc=20)
        tokenized_train = tokenized_train.filter(lambda x: x['len'] > 0)
        # concatenate all the ids in each dataset into one large file we can use for training
        _write_memmap(tokenized_train, filename)

if __name__ == '__main__':
    # Writing to postrain.bin with length 396749824
    tokenize_postrain()