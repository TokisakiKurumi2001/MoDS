from loguru import logger
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_1', type=str)
    parser.add_argument('--input_2', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()

    input_list = []
    hash_ls = []
    with open(args.input_1) as fin:
        for line in fin:
            _data = json.loads(line)
            # since there may be overlapped, a hash function is use to filter out duplicate
            hash_ls.append(hash(_data['prompt']))
            input_list.append(_data)

    num_duplicates = 0
    with open(args.input_2) as fin:
        for line in fin:
            _data = json.loads(line)
            line_hash = hash(_data['prompt'])
            if line_hash in hash_ls:
                num_duplicates += 1
                continue
            else:
                input_list.append(_data)

    logger.info(f'Number of duplicated iterms: {num_duplicates}')
    logger.info(f'Number of from both input files: {len(input_list)}')

    with open(args.output_file, 'w+') as fout:
        for d in input_list:
            fout.write(json.dumps(d) + "\n")