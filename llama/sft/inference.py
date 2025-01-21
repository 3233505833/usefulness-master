import transformers

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 加载合并后的分词器和模型
merged_model_path = ""# 假设你的合并模型已经保存在某个路径下
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
model = AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.float16).to("cuda")

model.eval()
sample = pd.read_json('./infer.json')

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

def query_model(system_message, input, temperature=0.1, max_length=1024):
    user_message =  input + "现在请对我提供的文档的有用性进行思考和打分。请先说出你的思考，然后再输出打分，格式为：思考：\n\n打分："
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_p=0.9,
        temperature=temperature,
        eos_token_id=terminators,
        max_length=max_length,
        max_new_tokens=300,
        return_full_text=False,
        pad_token_id=pipeline.model.config.eos_token_id
    )
    answer = sequences[0]['generated_text']
    return answer

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
output_file = 'evaluation_results.json'
if not os.path.isfile(output_file):
    with open(output_file, 'w') as f:
        json.dump([], f)

# 生成答案并实时写入 JSON
for idx, row in tqdm(sample.iterrows(), total=len(sample)):
    system_message = row["instruction"]
    question = row['input']
    answer = query_model(system_message, question, temperature=0.1, max_length=1024)
    print(answer)
    sample.at[idx, 'llama3_peft_answer'] = answer

    # 读取现有数据
    with open(output_file, 'r') as f:
        data = json.load(f)

    # 添加新结果
    row['llama3_peft_answer'] = answer
    data.append(row.to_dict())

    # 写入更新后的数据
    with open(output_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)