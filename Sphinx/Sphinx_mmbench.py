import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from PIL import Image
import math
from io import BytesIO
import base64

from SPHINX import SPHINXModel


all_options = ['A', 'B', 'C', 'D']

answerPrompt="Use the image and scene graph as context and answer the following question: "

sgPrompt='''

For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Object relationships that are relevant to answering the question.

Scene Graph:
'''


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

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def eval_model(args):
    # Model
    model = SPHINXModel.from_pretrained(pretrained_path="/home/chancharikm/compVL/LLaMA2-Accessory/SPHINX/SPHINX-v2-1k-weights", with_visual=True).to(device='cuda')

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
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question

            firstPrompt = question + sgPrompt
            #print(f'SG Prompt: {firstPrompt}')
            #print(f'qs {qs}')
            # if model.config.mm_use_im_start_end:
            #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            # else:
            #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            # if args.single_pred_prompt:
            #     if args.lang == 'cn':
            #         qs = qs + '\n' + "请直接回答选项字母。"
            #     else:
            #         qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model.generate_reponse( #No, this is not a typo. This seems to be typo inherited from the Sphinx codebase - let them know via Github!
                    [[firstPrompt, None]],
                    image,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_gen_len=256, 
                    seed=0)


            outputs = outputs.strip()

            sg = outputs

            secondPrompt = "Scene Graph: " + sg + '\n\n' + answerPrompt + qs + '\n' + "Answer with the option's letter from the given choices directly."
            #print(f'secondPrompt {secondPrompt}')

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model.generate_reponse( #No, this is not a typo. This seems to be typo inherited from the Sphinx codebase - let them know via Github!
                    [[secondPrompt, None]],
                    image,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_gen_len=256, 
                    seed=0)

            outputs = outputs.strip()
            #print(f'Final Output: {outputs}')

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": "Sphinx-v2-1k",
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
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
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
