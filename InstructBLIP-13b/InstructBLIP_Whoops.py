from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, InstructBlipConfig,  AutoModelForVision2Seq
import torch
from PIL import Image
import requests
from accelerate import init_empty_weights, infer_auto_device_map
import json
import os
from tqdm import tqdm
from datasets import load_dataset


hf_key=""  #Huggingface auth key
result_path=""  #Path to store result

sgPrompt='''
For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Object relationships that are relevant to answering the question.

Scene Graph:
'''


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


def get_ans(question, image_tensor, pred_sg=None):
    if pred_sg is None:
        prompt = "<Image> " + question + "\n\n" + sgPrompt
        max_token = 256
    else:
        prompt = f"<Image> Question:{question} Scene Graph:{pred_sg}\n\n Use the image and scene graph to reason and provide a short answer:"
        max_token = 64
    inputs = processor(images=image_tensor, text=prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=max_token,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=0.5,
        temperature=0,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    return generated_text


result_file = open(result_path, "w")
examples = load_dataset('nlphuji/whoops', use_auth_token=hf_key)

for item in tqdm(examples["test"]):

    image_tensor = item["image"].convert("RGB")
    all_pred = []
    for q_a in item["question_answering_pairs"]:
        

        question = q_a[0]
        pred_sg = get_ans(question, image_tensor)
        pred_ans = get_ans(question, image_tensor, pred_sg)
        all_pred.append(pred_ans)


    result_file.write(json.dumps({  "image_id": item["image_id"],
                                "question_answering_pairs": item["question_answering_pairs"],
                                "prediction": all_pred}) + "\n")


result_file.close()
