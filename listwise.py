
import numpy as np
from more_itertools import chunked
from utils.doc_use_tuple import get_doc_use_tuple,get_doc_use_tuple_from_KDD_json
import logging
from collections import defaultdict
logging.basicConfig(level=logging.ERROR)
import argparse
from yacs.config import CfgNode
from tqdm import tqdm
import json
from utils.new_data_loader import AbsDataloader
import math
from agents.data import Data
from utils.new_file_utils import load_jsonl, get_num_lines, check_file_and_mkdir_for_save, DataItem, save_jsonl, save_json,DataItem_new
from utils.new_data_prompt import GPT3FewShotSamplingPrompt
import os
from agents.recagent import Agent

class GPT3TextCLS(object):
    def __init__(self, config: CfgNode,log_interval: int = 50):
        self.config = config


        self.prompt = GPT3FewShotSamplingPrompt()
        self.dataloader = AbsDataloader()
        self.log_interval = log_interval
        os.environ["OPENAI_API_KEY"] = self.config["api_keys"][0]
        self.data = Data(self.config)
        self.model = Agent()
        import pandas as pd
        if self.config["data_name"] == "KDD":
            file_path = './data/KDD19zong_query_id.xlsx'
            self.df = pd.read_excel(file_path)


    def step1_prepare_input(self):
        save_data_path = os.path.join(self.config["save_log_dir"], "doc_data.jsonl")
        if self.config.get('data_name')=="KDD":
            test_items = [
                DataItem(*item) for item in get_doc_use_tuple_from_KDD_json([])
                if item[0] != ""
            ]
            batch_size = int(self.config['num_agents'])
            demonstration_candidates = [
                DataItem(*item) for item in get_doc_use_tuple_from_KDD_json([])
            ]
        else:

            test_items = [
                DataItem_new(*item) for item in get_doc_use_tuple([])
                if item[0] != ""
            ]
            batch_size = int(self.config['num_agents'])
            demonstration_candidates = [
                DataItem_new(*item) for item in get_doc_use_tuple([])
            ]


        if self.config.get('data_name')!="KDD":
            with open(save_data_path, "w", encoding='utf-8') as writer_f:
                for data_item_batch in tqdm(
                        chunked(test_items, batch_size),
                        total=math.ceil(len(test_items) / batch_size),
                        desc="step-1"
                ):
                    data_item_texts = [item.text for item in data_item_batch]
                    input_text_with_prompt = self.prompt.get_model_input_batch(
                        data_item_texts,
                        data_item_batch,
                        demonstrations_candidates=demonstration_candidates,
                        teacher_model=self
                    )
                    for item, prompt in zip(data_item_batch, input_text_with_prompt):
                        data_item_obj = {
                            "prompt_text": prompt,
                            "gold_label": item.label,
                            "text": item.text,
                            "query_str": item.query_str,
                            "task_id": item.task_id,
                            "query_lebal": item.query_lebal,
                            "user_id": item.user_id,
                            "query_id": item.query_id,
                            "serp_id": item.serp_id,
                            "order": item.order,
                            "dwell_time": item.dwell_time,
                            "his": item.his,"total_clicks_number":item.total_clicks_number,
    "clicked_ranks_list":item.clicked_ranks_list,"max_clicked_rank":item.max_clicked_rank,"avg_dwell_time":item.avg_dwell_time
                        }
                        writer_f.write(f"{json.dumps(data_item_obj, ensure_ascii=False)}\n")

        else:
            with open(save_data_path, "w", encoding='utf-8') as writer_f:
                for data_item_batch in tqdm(
                        chunked(test_items, batch_size),
                        total=math.ceil(len(test_items) / batch_size),
                        desc="step-1"
                ):
                    data_item_texts = [item.text for item in data_item_batch]
                    input_text_with_prompt = self.prompt.get_model_input_batch(
                        data_item_texts,
                        data_item_batch,
                        demonstrations_candidates=demonstration_candidates,
                        teacher_model=self
                    )
                    for item, prompt in zip(data_item_batch, input_text_with_prompt):
                        data_item_obj = {
                            "prompt_text": prompt,
                            "gold_label": item.label,
                            "text": item.text,
                            "query_str": item.query_str,
                            "task_id": item.task_id,
                            "query_lebal": item.query_lebal,
                            "user_id": item.user_id,
                            "query_id": item.query_id,
                            "serp_id": item.serp_id,
                            "order": item.order,
                            "dwell_time": item.dwell_time,
                            "his": item.his
                        }
                        writer_f.write(f"{json.dumps(data_item_obj, ensure_ascii=False)}\n")

            return save_data_path

        return save_data_path

    def doc_to_query_jsonl(self, query_save_path: str, step1_prompt_data_path: str, stage: str):
        saved_result_path = os.path.join(self.config["save_log_dir"], f"query{int(stage)}_input.jsonl")
        check_file_and_mkdir_for_save(saved_result_path, file_suffix=".jsonl")
        data_item_lst = load_jsonl(step1_prompt_data_path)

        batches = defaultdict(list)
        for item in data_item_lst:
            batches[int(item['query_id'])].append(item)

        all_batch_prompt_text = []
        def query_level_feature(item, df):
            search_id = item["query_id"]
            row = df.iloc[search_id]
            assert int(item["query_lebal"]) == int(row['Q_SAT'])

            if row.empty:
                return f""
            session_end_text = "这是整个搜索会话的最后一个查询。" if row['isSessionEnd'] else "这不是整个搜索会话的最后一个查询。"
            prompt = (
                f"[{int(item['order']) + 1}]"
                f"{item['text']}。该文档是搜索者第{int(item['order']) + 1}个点击进入阅读的文档，该文档在文档列表中的排名是第{int(item['serp_id']) + 1}名，"
                f"点击的文档的平均排名是 {round(row['AvgClickRank'], 1)}位。"
                f"点击的最深的文档是第 {int(round(row['ClickDepth']))}位。"
                f"信息搜索者对该文档的阅读时间为{np.ceil(item['dwell_time'] / 1000)}秒。"
                f"而搜索者在此查询中阅读每个文档的平均时间是 {round(row['AvgContent']/ 1000, 1)}秒."
            )
            return prompt

        def format_document_new(item):
            if len(item['clicked_ranks_list']) == 1:
                return (f"[{int(item['order']) + 1}]"+item['text'] +
                        "。该文档是搜索者在该查询中第" + str(int(item['order']) + 1) + "个点击进入阅读的文档。" +
                        "该文档在文档列表中的排名是第" + str(int(item['serp_id']) + 1) + "名。" +
                        "搜索者只点击了这一个文档就完成了搜索。" +
                        "搜索者对该文档的阅读时间为" + str(np.ceil(item['dwell_time'] / 1000)) + "秒。"
                        )
            else:
                return (f"[{int(item['order']) + 1}]"+item['text'] +
                        "。该文档是搜索者在该查询中第" + str(int(item['order']) + 1) + "个点击进入阅读的文档。" +
                        "该文档在文档列表中的排名是第" + str(int(item['serp_id']) + 1) + "名。" +
                        "而搜索者在此查询下总共点击了" + str(int(item['total_clicks_number'])) + "个文档，" +
                        "点击的所有文档的排名是" + str([x + 1 for x in item['clicked_ranks_list']]) + "。" +
                        "最大的点击深度是" + str(item['max_clicked_rank'] + 1) + "。" +
                        "搜索者对该文档的阅读时间为" + str(np.ceil(item['dwell_time'] / 1000)) + "秒。" +
                        "而搜索者在此查询上的所有文档的平均阅读时长是" + str(np.ceil(item['avg_dwell_time'] / 1000)) + "秒。"
                        )
        def generate_batch_prompt(batch_data_item, example_str="", stage=""):

            batch_prompt_text = [
                {"role": "system", "content": "你是信息检索领域的智能助手，能够根据信息搜索者的信息需求和查询对文档的有用性进行评分。"}
            ]
            if example_str:
                batch_prompt_text.append({"role": "user", "content": example_str})

            for i, item in enumerate(batch_data_item):
                if i == 0:
                    batch_prompt_text.append({"role": "user",
                                              "content": f"现在的信息需求和查询是:{item['prompt_text']}我会提供给你 {len(batch_data_item)} 个文档，每个文档都使用一个[数字]进行标识，请选择0个或者1个或者多个对回答这个查询最有用的文档。"})
                    batch_prompt_text.append({'role': 'assistant', 'content': '好的，请提供文档。'})
                if self.config["data_name"] == "KDD":
                    batch_prompt_text.append({'role': 'user', 'content': query_level_feature(item, self.df)})
                else:
                    batch_prompt_text.append({'role': 'user', 'content': format_document_new(item)})
                batch_prompt_text.append({'role': 'assistant', 'content': '已接收该文档。'})

            if stage == "4":
                batch_prompt_text.append({"role": "user",
                                          'content': "现在请选择对该查询最有用的0个1个或多个文档，即, 所选文档必须有效地满足信息需求，提供全面准确的信息，并具有帮助性、细节性、相关性、科普性和明确性，能显著提升信息搜索者的整体体验。所选文档应使用标识符输出，输出格式应为[][]，例如[1][2]；如果没有符合条件的文档，则输出[]。提示：你可以参考“有帮助”、“详细”、“相关”、“百科”、“具体”、“全面”这六个方面或其中几个方面进行考量，请先说出你的思考，然后再输出你选出的最有用的文档的结果，格式为：思考\n\n\n\n结果"})
            else:
                batch_prompt_text.append({"role": "user",
                                          'content': f"现在请先选择其中的0个或者1个或者多个文档打分为{stage}分。所选文档应使用标识符输出，输出格式应为[][]，例如[1][2]；如果没有符合条件的文档，则输出[]。提示：你可以参考“有帮助”、“详细”、“相关”、“百科”、“具体”、“全面”这六个方面或其中几个方面进行考量，请先说出你的思考，然后再输出你选出的最有用的文档的结果，格式为：思考\n\n\n\n结果。"})

            return batch_prompt_text

        if int(stage) == 4:
            for key_value, batch_data_item in tqdm(batches.items(), total=len(batches), desc="Processing Batches"):
                batch_prompt_text = generate_batch_prompt(batch_data_item, stage=stage)
                query_item = {
                    "prompt_text": batch_prompt_text,
                    "query_str": batch_data_item[0]["query_str"],
                    "task_id": batch_data_item[0]["task_id"],
                    "query_lebal": batch_data_item[0]["query_lebal"],
                    "user_id": batch_data_item[0]["user_id"],
                    "query_id": batch_data_item[0]["query_id"]
                }
                all_batch_prompt_text.append(query_item)
        else:
            example_item_lst = load_jsonl(query_save_path)

            def give_example(examples):
                example_str = "这里有一些在此查询和信息需求下的有用性分数为" + str(int(stage) + 1) + "分的文档的例子供你参考:<例子开始>"
                for idx, example in enumerate(examples, start=1):
                    letter = chr(96 + idx)
                    his = example["his"] if example["his"] else "None"
                    example_str += f"\n例子{letter}:{example['text']}该文档是搜索者第{int(example['order']) + 1}个点击进入阅读的文档，该文档在文档列表中的排名是第{int(example['serp_id']) + 1}名，信息搜索者对该文档的阅读时间为{np.ceil(example['dwell_time'] / 1000)}秒，在查看这个文档之前搜索者已经获取的信息有：{his}"
                return example_str + "<例子结束>"

            for key_value, batch_data_item in tqdm(batches.items(), total=len(batches), desc="Processing Batches"):
                examples = [example for example in example_item_lst if
                            example.get('query_id') == batch_data_item[0]['query_id']]
                example_str = give_example(examples)
                batch_prompt_text = generate_batch_prompt(batch_data_item, example_str=example_str, stage=stage)
                query_item = {
                    "prompt_text": batch_prompt_text,
                    "query_str": batch_data_item[0]["query_str"],
                    "task_id": batch_data_item[0]["task_id"],
                    "query_lebal": batch_data_item[0]["query_lebal"],
                    "user_id": batch_data_item[0]["user_id"],
                    "query_id": batch_data_item[0]["query_id"]
                }
                all_batch_prompt_text.append(query_item)

        save_jsonl(saved_result_path, all_batch_prompt_text)
        return saved_result_path

    def run_query_pre_results(self, stage, rater, step1_prompt_data_path: str, config_resume: bool = False):
        saved_result_path = os.path.join(self.config["save_log_dir"], "query" + str(int(stage)) + str(rater)+"_output.jsonl")
        if os.path.exists(saved_result_path) and config_resume:
            resume = True
        else:
            resume = False
        resume_offset = 0 if not resume else get_num_lines(saved_result_path)
        check_file_and_mkdir_for_save(saved_result_path, resume=resume, file_suffix=".jsonl")
        data_item_lst = load_jsonl(step1_prompt_data_path, offset=resume_offset)
        writer_mode = "w" if not resume else "a"
        writer_f = open(saved_result_path, writer_mode, encoding='utf-8')
        num_workers = int(self.config['num_agents'])

        for idx, batch_data_item in tqdm(enumerate(chunked(data_item_lst, num_workers)),
                                         total=math.ceil(len(data_item_lst) / num_workers), desc="step-2"):
            batch_prompt_text= []
            for i, item in enumerate(batch_data_item):
                batch_prompt_text.append(item["prompt_text"])
            api_keys = list(self.config['api_keys'])
            num_agents = int(self.config['num_agents'])
            gpt_returned_results = self.model.forward(self.config["model_name"], api_keys, num_agents,batch_prompt_text,)
            gpt_returned_text = gpt_returned_results
            gpt_returned_logprobs = len(gpt_returned_results) * [None]

            for data_item, returned_text, returned_logprobs, batch_prompt_text_inputgpt in zip(batch_data_item, gpt_returned_text,
                                                                   gpt_returned_logprobs,batch_prompt_text):
                save_result_item = {"gpt_returned_result": returned_text, "prompt_text": str(batch_prompt_text_inputgpt),
                                    "query_str": data_item["query_str"], "task_id": data_item["task_id"],
                                    "query_lebal": data_item["query_lebal"], "user_id": data_item["user_id"], "query_id": data_item["query_id"]
                                    }
                writer_f.write(f"{json.dumps(save_result_item,ensure_ascii=False)}\n")
        writer_f.close()
        return saved_result_path



    def query_jsonal_doc(self, refer_doc_path, output_query_path, output_update_query_path,
                         saved_result_doc_path,
                         stage):


        query_lst = load_jsonl(output_query_path)

        feasible = True

        if os.path.exists(saved_result_doc_path):
            raise FileExistsError(f"step1_map_competition_result_to_label -> {saved_result_doc_path}")
        update_data_item_lst = []


        error_data_item_lst = []
        doc_item_lst = load_jsonl(refer_doc_path)

        for query_item in query_lst:
            pred_label_temp = query_item["gpt_returned_result"]
            if pred_label_temp != None:
                query_item.update(
                    {"query_pred_label": pred_label_temp})
                update_data_item_lst.append(query_item)
            else:
                error_data_item_lst.append(query_item)

        num_workers = int(self.config['num_agents'])

        for idx, batch_data_item in tqdm(enumerate(chunked(error_data_item_lst, num_workers)),
                                         total=math.ceil(len(error_data_item_lst) / num_workers),
                                         desc="step-3"):
            restr1 = "你是信息检索领域的智能助手，能够根据信息搜索者的信息需求和查询对文档的有用性进行评分。你刚刚已经输出了以下的思考："
            restr2 = "接下来请你总结你刚刚已输出的思考直接输出你的选择，请你只输出一行回答所选的结果，不要说任何解释和思考，所选文档应使用标识符输出，输出格式应为[][]，例如[1][2]；如果没有符合条件的文档，则输出[]。"

            batch_prompt_text = [[{"role": "user", "content": restr1 + f'"{item["gpt_returned_result"]}"' + restr2}]
                                 for item in batch_data_item]
            api_keys = list(self.config['api_keys'])
            num_agents = int(self.config['num_agents'])
            gpt_returned_results2 = self.model.forward(self.config["model_name"], api_keys, num_agents,
                                                       batch_prompt_text, )
            for previous_item, gpt_returned_results2_str in zip(batch_data_item, gpt_returned_results2):
                previous_item["gpt_returned_result"] = previous_item[
                                                           "gpt_returned_result"] + "总结来说，" + gpt_returned_results2_str
                pred_label = (
                    gpt_returned_results2_str)
                previous_item.update(
                    {"query_pred_label": pred_label
                     })
                update_data_item_lst.append(previous_item)

        save_jsonl(output_update_query_path, update_data_item_lst)

        update_data_doc_item_lst = []
        for query_item in update_data_item_lst:
            pred_label = query_item["query_pred_label"]
            for doc_item in doc_item_lst:
                if doc_item["query_id"] == query_item["query_id"]:
                    if doc_item["order"] + 1 in pred_label:
                        doc_item.update(
                            {"pred_label": int(stage)
                             })
                        update_data_doc_item_lst.append(doc_item)
        if feasible:
            save_jsonl(saved_result_doc_path, update_data_doc_item_lst)

        return

    def update_0(self):

        unjudge1_path = os.path.join(self.config.save_log_dir, f"unjudge1.jsonl")
        judge0_path = os.path.join(self.config.save_log_dir, f"judge0.jsonl")
        doc_item_lst = load_jsonl(unjudge1_path)
        update_data_item_lst = []
        for doc_item in doc_item_lst:
            doc_item.update({"pred_label": 0})
            update_data_item_lst.append(doc_item)
        save_jsonl(judge0_path, update_data_item_lst)
    def update_1(self):

        unjudge2_path = os.path.join(self.config.save_log_dir, f"unjudge2.jsonl")
        judge1_path = os.path.join(self.config.save_log_dir, f"judge1.jsonl")
        doc_item_lst = load_jsonl(unjudge2_path)
        update_data_item_lst = []
        for doc_item in doc_item_lst:
            doc_item.update({"pred_label": 1})
            update_data_item_lst.append(doc_item)
        save_jsonl(judge1_path, update_data_item_lst)

    def merge_jsonl_file(self, file_list, output_file):
        combined_data = []

        for file_path in file_list:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    combined_data.append(json.loads(line))

        with open(output_file, 'w', encoding='utf-8') as file:
            for item in combined_data:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')


    def judge_and_refer_2_unjudge(self, refer_doc_path, judge_doc_path, stage):
        save_unjudge_result_path = os.path.join(self.config.save_log_dir, f"unjudge" + str(stage) + ".jsonl")
        unjudge_data_item_lst = []
        refer_list = load_jsonl(refer_doc_path)
        judge_list = load_jsonl(judge_doc_path)
        for refer_item in refer_list:
            find = 0
            for judge_item in judge_list:
                if judge_item["query_id"] == refer_item["query_id"] and judge_item["order"] == refer_item["order"]:

                    find=1
            if find==0:
                unjudge_data_item_lst.append(refer_item)
        save_jsonl(save_unjudge_result_path, unjudge_data_item_lst)
        return


def run():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", default=2333, type=int, help="random seed")
    parser.add_argument("--random", action="store_true", default=False)
    parser.add_argument("--config_path", default="../config/", type=str, help="path to the config file.")
    parser.add_argument("--test_file_name", default="test", type=str)
    parser.add_argument("--step_idx", default="", type=str)
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--fix_uid", default=False, action="")
    parser.add_argument("-c", "--config_file", type=str, default='config/config.yaml', help="Path to config file")
    args = parser.parse_args()

    step_idx_lst = [str(idx) for idx in args.step_idx.split("-")]
    config = CfgNode(new_allowed=True)
    config.merge_from_file(args.config_file)


    gpt3_for_text_cls_task = GPT3TextCLS(config=config)

    if "0" in step_idx_lst:
        gpt3_for_text_cls_task.step1_prepare_input()

    stages=["4","3","2","1"]
    if gpt3_for_text_cls_task.config.get('data_name')=="KDD":
        stages = ["4","3","2"]
    resume1temp = args.resume
    for stage in stages:
        if "1_"+str(stage) in step_idx_lst or "1" in step_idx_lst:
            if resume1temp == True:
                resume1temp = False
            else:
                if stage !="4":
                    judge_doc_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir, "judge"+str(int(stage)+1)+".jsonl")
                    unjudge_doc_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir, "unjudge"+str(int(stage)+1)+".jsonl")
                else:
                    judge_doc_path = ""
                    unjudge_doc_path =os.path.join(gpt3_for_text_cls_task.config.save_log_dir, "doc_data.jsonl")
                gpt3_for_text_cls_task.doc_to_query_jsonl(judge_doc_path, unjudge_doc_path, stage)
        if "2_"+str(stage) in step_idx_lst or "2" in step_idx_lst:
            step1_save_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir, "query"+str(int(stage))+"_input.jsonl")
            processes = []
            for rater in ["a", "b", "c","d", "e"]:
                import multiprocessing
                p = multiprocessing.Process(target=gpt3_for_text_cls_task.run_query_pre_results, args=(stage, rater, step1_save_path,True if args.resume else False,))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            if stage !="4":
                refer_doc_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir,"unjudge" + str(int(stage) + 1) + ".jsonl")
            else:
                refer_doc_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir, f"doc_data.jsonl")
            processes2 = []
            for rater in ["a", "b" , "c", "d", "e"]:
                output_query_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir,f"query" + str(int(stage)) + rater + "_output.jsonl")
                output_update_query_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir,f"query" + str(int(stage)) + rater + "_update_output.jsonl")
                saved_result_doc_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir,f"judge" + str(int(stage)) + rater + ".jsonl")

                p = multiprocessing.Process(target=gpt3_for_text_cls_task.query_jsonal_doc, args=(refer_doc_path, output_query_path, output_update_query_path,saved_result_doc_path, stage))
                processes2.append(p)
                p.start()
            for p in processes2:
                p.join()

            file_list = [os.path.join(gpt3_for_text_cls_task.config.save_log_dir,
                                                     f"judge" + str(int(stage)) + rater + ".jsonl") for rater in ["a", "b", "c","d", "e"]]
            judge_pre_file = os.path.join(gpt3_for_text_cls_task.config.save_log_dir, f"judge"+str(int(stage))+"_pre.jsonl")
            gpt3_for_text_cls_task.merge_jsonl_file(file_list, judge_pre_file)
            from utils.rater_group import group_5_rater
            judge_pre_file = os.path.join(gpt3_for_text_cls_task.config.save_log_dir, f"judge"+str(int(stage))+"_pre.jsonl")
            judge_file = os.path.join(gpt3_for_text_cls_task.config.save_log_dir, f"judge"+str(int(stage))+".jsonl")
            group_5_rater(judge_pre_file, judge_file, stage)
            gpt3_for_text_cls_task.judge_and_refer_2_unjudge(refer_doc_path, judge_file, stage)
    if "3" in step_idx_lst:
        if gpt3_for_text_cls_task.config.get('data_name')=="KDD":
            gpt3_for_text_cls_task.update_1()
        else:
            gpt3_for_text_cls_task.update_0()

if __name__ == "__main__":
    run()