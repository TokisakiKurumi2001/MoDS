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
    with open(args.input_1) as fin:
        for line in fin:
            _data = json.loads(line)
            input_list.append(_data)

    with open(args.input_2) as fin:
        for line in fin:
            _data = json.loads(line)
            input_list.append(_data)

    logger.info(f'Number of from both input files: {len(input_list)}')

    with open(args.output_file, 'w+') as fout:
        for d in input_list:
            fout.write(json.dumps(d) + "\n")