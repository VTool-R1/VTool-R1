import asyncio
import aiohttp
from tqdm.asyncio import tqdm
import re
import json
import os
import time

with open('judge_info.json', 'r') as file:
    data = json.load(file)
host = data.get('host')

if os.environ.get("LOCAL_JUDGE") == "YES":
    print("using LOCAL_JUDGE")
    url = f"http://{host}:7999/generate"
else:
    url = "http://grizzly-becoming-mantis.ngrok-free.app/generate"

with open("judge_prompt.txt", 'r') as file:
    judge_prompt = file.read()

batch_evaluates = []

num_test = 2000

for i in range(num_test):
    prompt = judge_prompt.replace("<question>", "In which category the index score value is 1?")
    prompt = prompt.replace("<gt>", "Legislators, senior officials and managers")
   
    prompt = prompt.replace("<answer>", "senior officials, managers and legislators")

    batch_evaluates.append(prompt)

async def fetch(session, prompt, semaphore):
    async with semaphore:
        payload = {
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 100
        }
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                return {"error": response.status, "text": await response.text()}

'''
async def main(prompts, semaphore):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, prompt, semaphore) for prompt in prompts]
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(prompts)):
            result = await f
            results.append(result)
        return results'''

async def main(prompts, semaphore):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, prompt, semaphore) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return results

def batch_process(batch):
    CONCURRENCY = 100  # Limit to avoid overwhelming the server
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = asyncio.run(main(batch, semaphore))
    return results

def extract_result(text):
    # Extract the last occurrence of content inside <>
    match = re.findall(r'<(.*?)>', text)

    if match:
        last_value = match[-1]
        return last_value == "|YES|"
    
    return False

# Run the event loop

correct = 0
incorrect = 0

print("Sending Request")

start_time = time.time()
results = batch_process(batch_evaluates)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.4f} seconds")

for result in results:
    response = result["text"][0]
    #print(response)
    #continue
    #print(extract_result(response))
    if extract_result(response):
        correct += 1
    else:
        incorrect += 1
    #response = result["text"][0]
    #print(extract_assistant_response(response))

print("CORRECT ", correct)
print("INCORRECT ", incorrect)