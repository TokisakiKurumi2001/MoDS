from loguru import logger
import json
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--threshold', type=float)

    quality_evaluation_list = []
    with open(args.input_file) as fin:
        for line in fin:
            _data = json.loads(line)
            quality_evaluation_list.append(_data)

    logger.info(f'number of input file: {len(quality_evaluation_list)}')

    threshold = args.threshold

    num_dict = {}
    all_num = len(quality_evaluation_list)

    result_json = []
    for item in tqdm(quality_evaluation_list):
        upper_num = math.ceil(item['reward_score'])
        lower_num = math.floor(item['reward_score'])
        num_dict[(lower_num, upper_num)] = num_dict.get((lower_num,upper_num),0) + 1
        if float(item['reward_score']) > threshold:
            result_json.append(item)

    print('The percent of each score interval:')
    for k, v in num_dict.items():
        print(str(k)+'  :  '+str(v)+'  '+str(float(v)/all_num))

    logger.info(f'number of data: {len(result_json)}')

    with open(args.output_file, 'w+') as fout:
        for d in result_json:
            fout.write(json.dumps(d) + "\n")