from loguru import logger
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()

    input_list = []
    with open(args.input_file) as fin:
        for line in fin:
            _data = json.loads(line)
            input_list.append(_data)

    logger.info(f'number of input file: {len(input_list)}')

    logger.info('Loading model ...')
    reward_name = "../model_ckpt/rank_model"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)

    device = torch.device('cuda:0')
    rank_model = rank_model.to(device)
    logger.success(f'Successfully loaded {reward_name} model')

    # testing
    question, answer = "Explain nuclear fusion like I am five", "Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants."
    inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
    score = rank_model(**inputs).logits[0].detach()
    print(float(score))

    result_list = []
    for element in tqdm(input_list):
        instruction = element['prompt']
        _input = ''
        _output = element['output']

        question = instruction
        answer = _output
        
        try:
            inputs = tokenizer(question, answer, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            score = rank_model(**inputs).logits[0].detach()
        except:
            print(instruction)
            print(_output)
            continue
        final_result = {'prompt':question,'output':answer,'reward_score':float(score)}
        result_list.append(final_result)

    logger.info(f'number of data: {len(result_list)}')

    with open(args.output_file, 'w+') as fout:
        for d in result_list:
            fout.write(json.dumps(d) + "\n")