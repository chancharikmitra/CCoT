from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, InstructBlipConfig,  AutoModelForVision2Seq
import torch
from PIL import Image
import requests
from accelerate import init_empty_weights, infer_auto_device_map
import json
import os
from tqdm import tqdm


# Determine if CUDA (GPU) is available.
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the model configuration.
config = InstructBlipConfig.from_pretrained("Salesforce/instructblip-vicuna-13b")

# Initialize the model with the given configuration.
with init_empty_weights():

    model = AutoModelForVision2Seq.from_config(config)
    model.tie_weights()

# Load the processor and model for image processing.
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b", device_map="auto")
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b",
                                                             device_map="auto")


sgPrompt='''
For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Object relationships that are relevant to answering the question.

Scene Graph:
'''

image_file = ""
question_path = ""
result_path = ""
result_file = open(result_path, 'w')


with open(question_path, 'r') as json_file:
    json_list = list(json_file)


for json_str in tqdm(json_list):


    result = json.loads(json_str)
    cur_image = image_file + result["image"] + ".png"
    image = Image.open(cur_image).convert("RGB")

    prompt = "<Image> Does the given caption accurately describe the given image? Caption:" +  result["caption"] + ".\n\n" + sgPrompt

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=0.5,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    

    answerPrompt = "Use the image and scene graph to reason and answer the question."
    prompt = "<Image> Question: Does the given caption accurately describe the given image? Caption:" +  result["caption"] + ". Scene Graph: " + generated_text + '\n\n' + answerPrompt

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=0.5,
        temperature=1,
    )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    stored_result = {"text":generated_text}


    result_file.write(json.dumps(stored_result) + "\n")
    result_file.flush()

result_file.close()
