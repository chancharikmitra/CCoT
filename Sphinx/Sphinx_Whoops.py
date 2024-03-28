import json
from tqdm import tqdm
from datasets import load_dataset
from SPHINX import SPHINXModel

result_path="" #Path to store the result
hf_token="" #Huggingface auth token


sgPrompt='''
For the provided image and its associated question, generate only a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question

Scene Graph:
'''
answerPrompt="\nUse the image and scene graph to reason and answer the question with a single phrase."


model = SPHINXModel.from_pretrained(pretrained_path="", with_visual=True)


def get_ans(question, pred_sg, image_tensor):

    final_ans = model.generate_response([[question + "Scene Graph:" + pred_sg + "\n\n" + answerPrompt, None]], image_tensor, max_gen_len=64, temperature=0)
    return final_ans


def get_sg(question, image):
    final_ans = model.generate_response([[question + sgPrompt, None]], image, max_gen_len=256, temperature=0)
    return final_ans


result_file = open(result_path, "w")
examples = load_dataset('nlphuji/whoops', use_auth_token=hf_token)
for item in tqdm(examples["test"]):
    
    image = item["image"]
    all_pred = []
    all_sg = []
    for q_a in item["question_answering_pairs"]:
        
        question = q_a[0]
        pred_sg = get_sg(question, image)
        pred_ans = get_ans(question, pred_sg, image)
        all_pred.append(pred_ans)

    result_file.write(json.dumps({  "image_id": item["image_id"],
                                "question_answering_pairs": item["question_answering_pairs"],
                                "prediction": all_pred}) + "\n")

result_file.close()
