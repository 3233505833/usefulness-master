import time
from typing import Any, Dict
import openai
from pydantic import BaseModel, Field
from yacs.config import CfgNode
from agents.data import Data
import json
from multiprocessing import Pool
from typing import Union, List, Tuple, Dict
from openai import OpenAI
import httpx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Agent(object):
    def generation(self, api_keys: list, id: int, model: str, messages, temperature, return_text) -> str:
        if model.startswith("gpt"):
            api_key = api_keys[id]
            client = OpenAI(
                base_url="",
                api_key=api_key,
                http_client=httpx.Client(
                    base_url="",
                    follow_redirects=True,
                ),
            )
            print(messages)
            print(len(messages))
            while True:
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=16380
                    )
                    break
                except Exception as e:
                    print(e)
                    if "This model's maximum context length is" in str(e):
                        print('reduce_length')
                        return 'ERROR::reduce_length'
                    time.sleep(0.1)
            if return_text:
                completion = completion.choices[0].message.content
            print("输出的全文为：")
            print(completion)
            return completion
        elif model=="llama-3":
            """
                    调用 Llama 模型进行推断，返回生成的文本答案。
                    api_keys: list, 可选，API 密钥列表，当前未用。
                    id: int, 当前任务的 ID。
                    model: str, 使用的模型名称。
                    messages: list, 含有系统消息和用户消息的列表。
                    temperature: float, 生成的温度，控制随机性。
                    """
            self.model_path = model_path
            self.device = device

            # 加载 tokenizer 和模型
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16).to(
                self.device)

            # 设置 pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device=0 if self.device == "cuda" else -1,  # cuda 设置
            )
            # 准备 prompt
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 设置终止符
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # 使用 pipeline 生成文本
            sequences = self.pipeline(
                prompt,
                do_sample=True,
                top_p=0.9,
                temperature=temperature,
                eos_token_id=terminators,
                max_length=1024,
                max_new_tokens=300,
                return_full_text=False,
                pad_token_id=self.model.config.eos_token_id
            )
            # 提取生成的文本
            answer = sequences[0]['generated_text']
            return answer
    def forward(self, model_name, api_keys: list, num_agents, prompt_lst) -> List[str]:
        """
        用来将prompt的一个batch列表给出输出。负责把一个batch的字符串得到结果.
        """
        if num_agents == 1 or len(prompt_lst) == 1:
            if isinstance(prompt_lst, list):
                prompt_input = prompt_lst[0]
            else:
                raise TypeError(prompt_lst)
            generate_result = self.generation(api_keys=api_keys, id=0, model=model_name, messages=prompt_input, temperature=0.7,return_text=True)
            return [generate_result]
        assert isinstance(prompt_lst, list)
        num_agents = min(num_agents, len(prompt_lst))
        pool_results = []
        pool = Pool(processes=num_agents)
        for worker_id in range(0, num_agents):
            result = pool.apply_async(self.generation,
                                      (api_keys, worker_id, model_name, prompt_lst[worker_id], 0.7, True))
            pool_results.append(result)  # 在这里直接进行并行化
        pool.close()
        pool.join()
        results = [item.get() for item in pool_results]
        return results

