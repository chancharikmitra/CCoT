import os
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
import math
import requests


api_key='' #Openai api key
question_path="" #MMbench tsv file path
result_path="" #Path to store result

sgPrompt='''
For the provided image and its associated question, generate only a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question
'''
answerPrompt="Use the image and scene graph as context and answer the following question: "
all_options = ['A', 'B', 'C', 'D']


def create_payload(cur_image, cur_text):
    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{cur_image}"
            }
            },
            {
            "type": "text",
            "text": cur_text
            }
        ]
        }
    ],
    "max_tokens": 512,
    "temperature":0
    }
  return payload


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

headers = {
"Content-Type": "application/json",
"Authorization": f"Bearer {api_key}"
}

questions = pd.read_table(question_path)
questions = get_chunk(questions, 1, 0)
result_file = open(result_path, "w")


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

            payload = create_payload(image, qs + sgPrompt)
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            cur_sg = response.json()["choices"][0]["message"]["content"]


            new_q = f"Scene Graph: {cur_sg}\n\n{answerPrompt}{qs}\nAnswer with the option's letter from the given choices directly."
            payload = create_payload(image, new_q)
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            final_ans = response.json()["choices"][0]["message"]["content"]


            temp_result = {"question_id":idx, "text":final_ans}
            result_file.write(json.dumps(temp_result) + "\n")


            is_done = True
            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
        except:
            fail_count += 1
            if fail_count == 5:
                break
            is_done = False
result_file.close()
