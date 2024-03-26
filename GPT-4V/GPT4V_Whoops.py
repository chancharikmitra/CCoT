import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import requests
from PIL import Image
import math
from datasets import load_dataset
import base64
from io import BytesIO


result_path="" ##path to store result
api_key="" #Openai api key
hf_key="" #Huggingface auth key


sgPrompt='''
For the provided image and its associated question, generate only a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question
'''
answerPrompt=".Use the image and scene graph as context and answer the following question: "


def get_ans(question, image_tensor, pred_sg=None):
    if pred_sg is None:
        cur_prompt = question + sgPrompt
        max_tokens = 512
    else:
        cur_prompt = "Scene Graph: " + pred_sg + answerPrompt + question
        max_tokens = 128

    buffered = BytesIO()
    image_tensor.convert('RGB').save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    

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
                "url": f"data:image/jpeg;base64,{img_str}"
            }
            },
            {
            "type": "text",
            "text": cur_prompt
            }
        ]
        }
    ],
    "max_tokens": max_tokens,
    "temperature":0
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    cur_ans = response.json()["choices"][0]["message"]["content"]
    return cur_ans
    

result_file = open(result_path, "w")
examples = load_dataset('nlphuji/whoops', use_auth_token="")
for item in tqdm(examples["test"]):
    
    is_done = False
    fail_count = 0

    while not is_done:
        try:
            image = item["image"]

            all_pred = []
            for q_a in item["question_answering_pairs"]:
                
                question = q_a[0]
                pred_sg = get_ans(question, image)
                pred_ans = get_ans(question, image, pred_sg)
                all_pred.append(pred_ans)


            result_file.write(json.dumps({  "image_id": item["image_id"],
                                        "question_answering_pairs": item["question_answering_pairs"],
                                        "prediction": all_pred}) + "\n")
            is_done = True
        except:
            fail_count += 1
            if fail_count == 5:
                break
            is_done = False

result_file.close()
