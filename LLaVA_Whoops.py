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
answerPrompt="\nUse the image and scene graph to reason and answer the question with a single phrase."


disable_torch_init()
model_path = os.path.expanduser("liuhaotian/llava-v1.5-13b")
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)


def get_ans(question, image_tensor, sg = 0):

    if sg == 1:
        new_token = 256
        qs = DEFAULT_IMAGE_TOKEN + question  + sgPrompt
    else:
        new_token = 64
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
            max_new_tokens=new_token,
            use_cache=True)


    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    return outputs


ans_file = open("", "w")
examples = load_dataset('nlphuji/whoops', use_auth_token="")
for item in tqdm(examples["test"]):


    image_tensor = process_images([item["image"]], image_processor, model.config)[0]
    all_pred = []
    for q_a in item["question_answering_pairs"]:
        
        question = q_a[0]
        pred_ans = get_ans(question, image_tensor, sg = 1)

        
        question = q_a[0] + "Scene Graph:" +  pred_ans + "\n\n" + answerPrompt
        pred_ans = get_ans(question, image_tensor)
        all_pred.append(pred_ans)

    ans_file.write(json.dumps({  "image_id": item["image_id"],
                                "question_answering_pairs": item["question_answering_pairs"],
                                "prediction": all_pred}) + "\n")
    ans_file.flush()

ans_file.close()
