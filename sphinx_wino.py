from SPHINX import SPHINXModel
from PIL import Image
import torch
import json
from tqdm import tqdm

sgPrompt='''
For the provided image and its associated question, generate only a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question
'''

model = SPHINXModel.from_pretrained(pretrained_path="", with_visual=True)


image_file = ""
question_path = ""
result_path = ""
result_file = open(result_path, 'w')


with open(question_path, 'r') as json_file:
    json_list = list(json_file)


for json_str in tqdm(json_list):

    cur_pair = json.loads(json_str)
    cur_image = Image.open(image_file + cur_pair["image"] + ".png")
    cur_caption = cur_pair["caption"]

    cur_question = [["Does the given caption accurately describe the given image? Caption:" + cur_caption + ".\n\n" + sgPrompt, None]]
    cur_sg = model.generate_response(cur_question, cur_image, max_gen_len=256, temperature=0)

    new_question = [["Does the given caption accurately describe the given image? Caption:" +  cur_caption + ".\n\nScene graph:" + cur_sg + '''\n\nBased on the image and scene graph, provide a explanation to the answer.''', None]]
    final_ans = model.generate_response(new_question, cur_image, max_gen_len=256, temperature=0)



    stored_response = {"text":final_ans}
    result_file.write(json.dumps(stored_response) + "\n")
    result_file.flush()


result_file.close()
