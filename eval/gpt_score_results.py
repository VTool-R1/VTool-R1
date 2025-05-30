from io import BytesIO
import random
import re
import json
import datetime
import argparse

from tqdm import tqdm
import numpy as np
from PIL import Image

from async_gpt_utils import get_completion_list
from verl.tooluse.chart_data import *


def main():
    parser = argparse.ArgumentParser(description="Use GPT and Exact Match to score model output. VLMToolUse.")

    # Add arguments
    parser.add_argument("-i", "--input_file", help="Path to model results", default=None)
    parser.add_argument("-og", "--out_gpt_score_file", help="Path to the output file for gpt scores", default="gpt_score_output.jsonl")
    parser.add_argument("-oe", "--out_exact_score_file", help="Path to the output file for exact match scores", default="exact_score_output.jsonl")

    
    # Parse arguments
    args = parser.parse_args()
    
    if args.input_file is None:
        print("No input_file given! Please provide path to model results.")
        return 1

    # store stats
    gpt_scores = []

    """
    Eval Loop
    """
    predicts = []
    print("Eval...")
    with open(args.input_file, 'r') as in_f:
        for idx, line in enumerate(in_f.readlines()):
            result_obj = json.loads(line)
            predicts.append(result_obj)


    print("Using gpt to score model predicts...")
    for result in tqdm(predicts):
        gpt_score, prediction = compute_acc_from_raw_answer(result['query'], result['ground_truth'], result['model_response'])
        with open(args.out_gpt_score_file, 'a') as f:
            f.write(str(gpt_score) + '\n')
        gpt_scores.append(gpt_score)

    print('GPT Scored accuracy: ', len(list(filter(lambda x: x == 1, gpt_scores))) / len(gpt_scores))

    ### batch eval result
    correct_or_not = []
    for idx, result in enumerate(predicts):
        gt = result['ground_truth']
        model_response_text = result['model_response']
        re_with_terminate = r'FINAL ANSWER:\s*(.*?)\s*TERMINATE'
        re_without_terminate = r'FINAL ANSWER:\s*(.*?)(?=\.\s|\.?$)'
        model_response_choice = re.findall(re_with_terminate, model_response_text)
        if len(model_response_choice) == 0:
            print('re.findall() failed')
            correct_or_not.append(False)
        else:
            model_response_choice = model_response_choice[-1]
            if model_response_choice.lower() == gt.lower():
                print(f'correct: {gt}')
                correct_or_not.append(True)
            else:
                print(f'wrong: {model_response_choice} | gt: {gt}')
                correct_or_not.append(False)
    with open(args.out_exact_score_file, 'a') as f:
        for each in correct_or_not:
            f.write('1\n' if each else '0\n')

    print("Mean Overall Exact Match Score:", sum(correct_or_not) / len(correct_or_not))
    
    print("Scoring Complete.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)