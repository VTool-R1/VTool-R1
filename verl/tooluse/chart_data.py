import json
import os
from openai import OpenAI
import time
import re

def match_score(prompt):
    retry_limit = 10
    eval_model = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for retry in range(retry_limit):
        try:
            response = eval_model.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            score = response.choices[0].message.content
            return int(score.strip())
        except Exception as e:
            print(e)
            time.sleep(1)
    return 'Failed'

def compute_acc_single(ex, pred):
    pred = pred.replace('||', '|')
    solution = ex["answer"]
    # Use regex to find text between 'FINAL ANSWER:' and 'TERMINATE'
    result = re.search(r'FINAL ANSWER:\s*(.*?)\s*TERMINATE', pred)

    # Extract the matched text if found
    prediction = result.group(1) if result else pred
    answer_score = 0
    grading_query = f'''
    Rate my prediction given the correct answer.
    Prediction: {prediction}
    Answer: {solution}
    If you think the prediction is correct, return 1, otherwise return 0. Return 0 or 1 only.'''
    answer_score = match_score(grading_query)
    return answer_score, prediction, solution


def compute_acc_from_raw_answer(question, solution, pred):
    pred = pred.replace('||', '|')
    result = re.search(r'FINAL ANSWER:\s*(.*?)\s*TERMINATE', pred)

    # Extract the matched text if found
    prediction = result.group(1) if result else pred
    answer_score = 0
    grading_query = f'''
    You are given a prediction for the question. You need to rate it given the correct answer. Disregard the format, and only rate based on the content. If you think the prediction is correct, i.e. same as the correct answer, then return 1, otherwise return 0. Return 0 or 1 only.

    # Example
    # Question: What is the height of the tower?
    # Prediction: ANSWER: The tower is built in China from 200 years ago. The total hight of the tower is 180 Meters. TERMINATE
    # Correct Answer: 80 Meters
    # Your Response: 0

    # Example
    # Question: What is the difference?
    # Prediction: ANSWER: 69%. TERMINATE
    # Correct Answer: 69
    # Your Response: 1

    # Example
    # Question: What is the increase?
    # Prediction: ANSWER: 3%. TERMINATE
    # Correct Answer: 0.03
    # Your Response: 1

    # Example
    # Question: What is the ratio?
    # Prediction: ANSWER: 1.541. TERMINATE
    # Correct Answer: 1.54
    # Your Response: 1

    # Question: {question}
    # Prediction: {prediction}
    # Correct Answer: {solution}
    # Your Response:'''
    # print(grading_query)
    answer_score = match_score(grading_query)
    return answer_score, prediction
