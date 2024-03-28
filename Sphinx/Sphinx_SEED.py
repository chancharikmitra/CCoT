import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from SPHINX import SPHINXModel
import math

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


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, sg_prompt):#, ans_folder: str='./playground/data/eval/MME/answers/llava-v1.5-13b.jsonl'):
        self.questions = questions
        self.image_folder = image_folder
        self.sg_prompt = sg_prompt

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        if self.sg_prompt == 1:
            lst = line['text'].split('\n')
            #lst.pop()
            #qs = '\n'.join(lst)
            qs = lst[0] + sgPrompt
         
        else:
            qs = line["text"]
         

        return qs, self.image_folder, image_file

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, batch_size=1, num_workers=4, sg_prompt = 0):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, sg_prompt)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def get_sg_prompt(args):
    
    model = SPHINXModel.from_pretrained(pretrained_path="/home/chancharikm/compVL/LLaMA2-Accessory/SPHINX/SPHINX-v2-1k-weights", with_visual=True).to(device='cuda')
    


    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]


    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)


    q_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(q_file), exist_ok=True)
    q_file = open(q_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, sg_prompt = 1)


    for (raw_text, image_folder, image_file), line in tqdm(zip(data_loader, questions), total=len(questions)):
        #No idea why raw_text, image_folder, image_file are being unpacked as tuples (value, __blank__), but I'm going with it for now...
        #It's fine in the data_loader but not here: weird, I suspect it has to do with this misaligned tqdm

        image = Image.open(os.path.join(image_folder[0], image_file[0])).convert('RGB')
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model.generate_reponse( #No, this is not a typo. This seems to be typo inherited from the Sphinx codebase - let them know via Github!
                [[raw_text[0], None]],
                image,
                temperature=args.temperature,
                top_p=args.top_p,
                max_gen_len=256, 
                seed=0)
        outputs = outputs.strip()


        

        q_file.write(json.dumps({  "image": line["image"],
                                   "text": "Scene Graph: " + outputs + '\n\n' + answerPrompt + line["text"],
                                   "category": line["category"],
                                   "question_id": line["question_id"]}) + "\n")

       
    q_file.close()


def eval_model(args):

    model = SPHINXModel.from_pretrained(pretrained_path="/home/chancharikm/compVL/LLaMA2-Accessory/SPHINX/SPHINX-v2-1k-weights", with_visual=True).to(device='cuda')
    print(f'Chunk ID {args.chunk_idx} Loaded!')
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]



    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)


    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")



    data_loader = create_data_loader(questions, args.image_folder)

    for (raw_text, image_folder, image_file), line in tqdm(zip(data_loader, questions), total=len(questions)):
        #No idea why raw_text, image_folder, image_file are being unpacked as tuples (value, __blank__), but I'm going with it for now...
        #It's fine in the data_loader but not here: weird, I suspect it has to do with this misaligned tqdm
        idx = line["question_id"]
        cur_prompt = line["text"]
        model_name = "Sphinx-v2-1k"
        image = Image.open(os.path.join(image_folder[0], image_file[0])).convert('RGB')
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model.generate_reponse(
                [[raw_text[0], None]],
                image,
                temperature=args.temperature,
                top_p=args.top_p,
                max_gen_len=256, 
                seed=0)

        outputs = outputs.strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
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

    parser.add_argument("--scene_graph", type=int, default=0)

    args = parser.parse_args()

    if args.scene_graph == 0:
        eval_model(args)
    else:
        get_sg_prompt(args)
