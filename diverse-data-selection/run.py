import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from kcenter_greedy import *
from tqdm import tqdm
import argparse
from loguru import logger

MAX_PROMPT_LENGTH=860

@torch.no_grad()
def bert_embedding(texts,batch=100):
    tokenizer = AutoTokenizer.from_pretrained('../model_ckpt/bert-base-uncased')
    model = AutoModel.from_pretrained('../model_ckpt/bert-base-uncased')
    device = torch.device('cuda:0')
    model = model.to(device)
    cls_hid_li = []
    for i in tqdm(range(0, len(texts), batch), desc="Encoding", unit='batch'):
        batch_sample = texts[i:i+batch]
        inputs = tokenizer(batch_sample, return_tensors='pt', truncation=True, padding=True, max_length=MAX_PROMPT_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        last_hids = model(**inputs).last_hidden_state
        cls_hids = last_hids[:, 0, :].squeeze()
        cls_hid_li.append(cls_hids)
        i += batch
    
    cls_hids_tensor = torch.concat(cls_hid_li, dim=0)
    np.save("bert_embedding.npy",cls_hids_tensor.cpu())
    return np.array(cls_hids_tensor.cpu())

def sample_func(text_list,K):
    result = []
    if os.path.exists("bert_embedding.npy"):
        text_embedding = np.load("bert_embedding.npy")
    else:
        text_embedding = bert_embedding(text_list)
        np.save("bert_embedding.npy",text_embedding)
    
    result = []

    k_center = kCenterGreedy(text_embedding)
    
    already_selected = None
    result = k_center.select_batch_(text_embedding,already_selected,K)
    return result

def main(input_file, output_file, K):
    data = []
    with open(input_file) as fin:
        for line in fin:
            _data = json.loads(line)
            data.append(_data)

    logger.info(f'There are {len(data)} samples in {input_file}')

    instruction_list = []
    for d in data:
        instruction_list.append(d["prompt"])
    res = sample_func(text_list = instruction_list, K = K)
    logger.success(f"Successfully sample {len(res)}.")

    data_li = []
    for index in res:
        data_li.append(data[index])
    with open(output_file, 'w+') as fout:
        for d in data_li:
            fout.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--top-k', type=int)
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.top_k)
