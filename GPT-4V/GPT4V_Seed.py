from PIL import Image
import base64
import requests
import json
from tqdm import tqdm
from openai import OpenAI
import json

image_file = '' # Image Path
question_path = "" #Question Path
result_path = "" #File to store result
result_file = open(result_path, 'w')
api_key='' #Openai api key

sgPrompt='''
For the provided image and its associated question, generate only a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question
'''


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


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


with open(question_path, 'r') as json_file:
    json_list = list(json_file)

headers = {
"Content-Type": "application/json",
"Authorization": f"Bearer {api_key}"
}


for json_str in tqdm(json_list):
    is_done = False
    fail_count = 0

    while not is_done:
        try:
            result = json.loads(json_str)


            cur_image = encode_image(image_file + result["image"])
            cur_question = result["text"]
            cur_id = result["question_id"]

            payload = create_payload(cur_image, "Question: " + cur_question.split("?")[0] + "?" + sgPrompt)
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            cur_sg = response.json()["choices"][0]["message"]["content"]

            new_q = f"Scene Graph: {cur_sg}. Use the image and scene graph as context and answer the following question: {cur_question}"
            payload = create_payload(cur_image, new_q)

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            final_ans = response.json()["choices"][0]["message"]["content"]


            temp_result = {"question_id":cur_id, "text":final_ans}
            result_file.write(json.dumps(temp_result) + "\n")
            is_done = True
        except:
            fail_count += 1
            if fail_count == 5:
                break
            is_done = False

result_file.close()
