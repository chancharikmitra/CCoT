import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.utils import disable_torch_init
from llava.mm_utils import load_image_from_base64
from PIL import Image
import math
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, InstructBlipConfig,  AutoModelForVision2Seq
import torch
from PIL import Image
import requests
from accelerate import init_empty_weights, infer_auto_device_map
import json
import os


processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b", device_map="auto")
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b")
model.to("cuda:1")


answerPrompt="Use the image and scene graph as context and answer the following question: "

sgPrompt='''
For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question

Scene Graph:
'''

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


def eval_model(args):
    # Model
    disable_torch_init()

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']

            question = row["question"]

            image = load_image_from_base64(row['image'])

            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question


            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:

                    qs = qs + sgPrompt


            prompt = "<Image> " + qs 
            inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:1")
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


            prompt_score = "<Image>" + " Scene Graph: " + generated_text + '\n\n' + answerPrompt + cur_prompt + ". The correct letter is"
            inputs2 = processor(images=image, text=prompt_score, return_tensors="pt").to("cuda:1")
            outputs2 = model.generate(
                **inputs2,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=0.5,
                temperature=1,
            )

            generated_text = processor.batch_decode(outputs2, skip_special_tokens=True)[0].strip()

            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": generated_text,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")

    args = parser.parse_args()

    eval_model(args)
