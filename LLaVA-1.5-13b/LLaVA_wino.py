import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from datasets import load_dataset


sgPrompt='''
For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question

Scene Graph:
'''


disable_torch_init()
model_path = os.path.expanduser("liuhaotian/llava-v1.5-13b")
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)


def get_ans(question, image_tensor, sg = 0):

    if sg == 1:
        qs = DEFAULT_IMAGE_TOKEN + question  + sgPrompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + question 

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = torch.unsqueeze(tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'), 0)
    input_ids = input_ids.to(device='cuda', non_blocking=True)
    
    image_tensor = torch.unsqueeze(image_tensor, 0)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            do_sample=True if 0 > 0 else False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True)


    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    return outputs



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

    prompt = "Does the given caption accurately describe the given image? Caption:" +  result["caption"] + ".\n\n" + sgPrompt

    cur_sg = get_ans(prompt, image, sg=1)


    answerPrompt = "Use the image and scene graph to reason and answer the question."
    prompt = "Question: Does the given caption accurately describe the given image? Caption:" +  result["caption"] + ". Scene Graph: " + cur_sg + '\n\n' + answerPrompt

    final_ans = get_ans(prompt, image)
    stored_result = {"text":final_ans}
    result_file.write(json.dumps(stored_result) + "\n")
    result_file.flush()

result_file.close()
