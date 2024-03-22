import os
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
import math
import requests


api_key=''
sgPrompt='''
For the provided image and its associated question, generate only a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question
'''
answerPrompt="Use the image and scene graph as context and answer the following question: "
all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


questions = pd.read_table(os.path.expanduser(""))
questions = get_chunk(questions, 1, 0)
answers_file = os.path.expanduser("")
os.makedirs(os.path.dirname(answers_file), exist_ok=True)
ans_file = open(answers_file, "w")


for index, row in tqdm(questions.iterrows(), total=len(questions)):
    is_done = False
    fail_count = 0

    while not is_done:
        try:
            options = get_options(row, all_options)
            cur_option_char = all_options[:len(options)]


            idx = row['index']
            question = row["question"]
            image = row['image']


            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = question


            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }


            payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                    },
                    {
                    "type": "text",
                    "text": qs + "/n" + sgPrompt
                    }
                ]
                }
            ],
            "max_tokens": 512
            }


            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            cur_sg = response.json()["choices"][0]["message"]["content"]


            new_q = "Scene Graph: " + cur_sg + "/n/n" + answerPrompt + qs + '\n' + "Answer with the option's letter from the given choices directly."
            payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                    },
                    {
                    "type": "text",
                    "text": new_q
                    },
                ]
                }
            ],
            "max_tokens": 512
            }


            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            final_ans = response.json()["choices"][0]["message"]["content"]


            q0 = {}
            q0["question_id"] = idx
            q0["text"] = final_ans
            ans_file.write(json.dumps(q0) + "\n")


            ans_file.flush()
            is_done = True
            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
        

        except:
            fail_count += 1
            if fail_count == 5:
                break
            is_done = False
ans_file.close()
