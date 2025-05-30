# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def is_number(s):
    try:
        float(s)  # Will handle both int and float strings
        return True
    except ValueError:
        return False
    
def similarity_score(a, b):
    if a == b:
        return 1.0
    if a == 0 or b == 0:
        return 0.0
    return 1 - (abs(a - b) / max(abs(a), abs(b)))

def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        '''predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )'''
        answers = re.findall(r'FINAL ANSWER:\s*(.*?)(?=\.\s|\.?$)', predict)
        
        overall_score = 0

        sub_gts = ground_truth.split("|||")

        if len(answers) > 0:
            answers = answers[0]

            expected_answers = len(sub_gts)
            correct_answers = 0

            for answer in answers.split('||'):
                
                candidate_scores = []

                for gt in sub_gts:
                    if is_number(gt) and is_number(answer):
                        gt = float(gt)
                        ans = float(answer)

                        candidate_scores.append(similarity_score(gt, ans))

                    else:
                        if answer == gt:
                            candidate_scores.append(1)
                            #correct_answers += 1

                if len(candidate_scores) > 0:
                    correct_answers += max(candidate_scores)

            if expected_answers == 0:
                expected_answers = 1

            overall_score = correct_answers/expected_answers

        scores.append(
            {
                "overall": overall_score
            }
        )

    return scores
