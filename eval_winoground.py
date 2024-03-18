from openai import OpenAI
import time
import json


client = OpenAI(
    api_key='',
)


def generate_response(prompt):
    # Call the OpenAI API to generate a response
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {'role': 'user', 'content': prompt}
            ],
        max_tokens=512,
        n=1,
        temperature=0,
        top_p=0.1,
        frequency_penalty=0.0,
        presence_penalty=0,
    )
    # Get the response text from the API response
    response_text = response.choices[0].message.content

    return response_text

#Text score. Identify which caption-explanation paire matches better.
def get_ans(caption1, caption2, exp1, exp2):
    #original
    PROMPT = "Caption A:" + caption1 + ". Explanation A:" + exp1 + "\n\nCaption B:" + caption2 + ". Explanation B:" + exp2 + "\n\n  Each explanation tries to justify why an image match with the corresponding caption. Pick the most logical explanation and return only an alphabet letter."
    return generate_response(PROMPT)

#Image score. Identify which explanation match better with the caption.
def get_ans2(caption1, caption2, exp1, exp2):
    PROMPT = "Caption:" + caption1 + ".Explanation A:" + exp1 + "Explanation B:" + exp2 + "\n\n Pick the explanation with information that align with the caption and return only an alphabet letter."
    return generate_response(PROMPT)


ans_path = ""
with open(ans_path, 'r') as json_file:
    json_list = list(json_file)
qs_path = ""
with open(qs_path, 'r') as q_file:
    q_list = list(q_file)


count = 0
correct_text = 0
correct_img = 0
correct_group = 0


print("begin")
while count < 1600:
    
    result1 = json.loads(json_list[count])
    result2 = json.loads(json_list[count+1])
    result3 = json.loads(json_list[count+2])
    result4 = json.loads(json_list[count+3])

    cap1 = json.loads(q_list[count])["caption"]
    cap2 = json.loads(q_list[count+1])["caption"]

    expain1 = result1["answer"]
    expain2 = result2["answer"]
    expain3 = result3["answer"]
    expain4 = result4["answer"]

    #Get text score
    text_result1 = get_ans(cap1, cap2, expain1, expain2)
    text_result2 = get_ans(cap1, cap2, expain3, expain4)

    #Get image score
    text_result3 = get_ans2(cap1, cap1, expain1, expain3)
    text_result4 = get_ans2(cap2, cap2, expain2, expain4)


    if (text_result1 == "A" and text_result2 == "B"):
        correct_text += 1
    if (text_result3 == "A" and text_result4 == "B"):
        correct_img += 1
        
    if (text_result1 == "A" and text_result2 == "B") and (text_result3 == "A" and text_result4 == "B"):
        correct_group += 1
    count += 4

print("text score:", correct_text / 400)
print("image score:", correct_img / 400)
print("group score:", correct_group / 400)
