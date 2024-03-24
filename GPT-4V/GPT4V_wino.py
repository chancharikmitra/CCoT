from PIL import Image
import base64
import requests
import json
from tqdm import tqdm
from openai import OpenAI
import json

image_file = "" # Image Path
question_path = ""
result_path = ""
result_file = open(result_path, 'w')
api_key=""

sgPrompt='''
For the provided image and its associated question, generate only a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question
'''
answerPrompt=".Use the image and scene graph as context and answer the following question: "


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


with open(question_path, 'r') as json_file:
    json_list = list(json_file)


for json_str in tqdm(json_list):
    is_done = False
    fail_count = 0

    while not is_done:
        try:
            result = json.loads(json_str)

            cur_image = encode_image(image_file + result["image"] + ".png")
            cur_caption = result["caption"]
            cur_question = "Does the given caption accurately describe the given image? Caption:" + cur_caption + ".\n\n" + sgPrompt


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
                        "url": f"data:image/jpeg;base64,{cur_image}"
                    }
                    },
                    {
                    "type": "text",
                    "text": cur_question
                    }
                ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            cur_sg = response.json()["choices"][0]["message"]["content"]
            new_q = "Question: Does the given caption accurately describe the given image? Caption:" +  cur_caption + ".\n\nAnswer: Scene graph:" + cur_sg + '''\n\nUse the scene graph and image to reason and answer the question:'''


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
                    "text": new_q
                    },
                ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            final_ans = response.json()["choices"][0]["message"]["content"]


            stored_response = {"text":final_ans}
            result_file.write(json.dumps(stored_response) + "\n")
            result_file.flush()
            is_done = True
        except:

            fail_count += 1
            if fail_count == 5:
                break
            is_done = False

result_file.close()
