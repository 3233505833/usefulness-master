
import numpy as np
from more_itertools import chunked
from utils.doc_use_tuple import get_doc_use_tuple,get_doc_use_tuple_from_KDD_json
import logging
import copy
from collections import defaultdict
import pandas as pd
logging.basicConfig(level=logging.ERROR)
import argparse
from yacs.config import CfgNode
from tqdm import tqdm
import os
from sklearn.metrics import mean_absolute_error
import json
from utils.new_data_loader import AbsDataloader
import math
from agents.data import Data
from utils.new_file_utils import load_jsonl, get_num_lines, check_file_and_mkdir_for_save,  save_jsonl, save_json, DataItem,DataItem_new
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score

from utils.new_data_prompt import GPT3FewShotSamplingPrompt
import os
from typing import List
import os
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, cohen_kappa_score
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from agents.recagent import Agent
import re



class GPT3TextCLS(object):
    def __init__(self, config: CfgNode):
        self.config = config
        self.prompt = GPT3FewShotSamplingPrompt()
        self.dataloader = AbsDataloader()
        os.environ["OPENAI_API_KEY"] = self.config["api_keys"][0]
        self.data = Data(self.config)
        self.model = Agent()

    def step1_prepare_input(self, resume: bool):
        save_data_path = os.path.join(self.config["save_log_dir"], "step1_data.jsonl")
        check_file_and_mkdir_for_save(save_data_path, resume=resume, file_suffix=".jsonl")
        if self.config.get('data_name')=="KDD":
            test_items = [
                DataItem(*item) for item in get_doc_use_tuple_from_KDD_json([])
                if item[0] != ""
            ]
            demonstration_candidates = [
                DataItem(*item) for item in get_doc_use_tuple_from_KDD_json([])
            ]
        else:

            test_items = [
                DataItem_new(*item) for item in get_doc_use_tuple([])
                if item[0] != ""
            ]
            demonstration_candidates = [
                DataItem_new(*item) for item in get_doc_use_tuple([])
            ]
        test_items = test_items

        writer_mode = "w" if not resume else "a"
        writer_f = open(save_data_path, writer_mode, encoding='utf-8')
        batch_size = int(self.config['num_agents'])
        for idx, data_item in tqdm(enumerate(chunked(test_items, batch_size)),total=math.ceil(len(test_items) / batch_size), desc="step-1"):

            data_item_text = [item.text for item in data_item]
            input_text_with_prompt = self.prompt.get_model_input_batch(data_item_text,data_item,
                                                                       demonstrations_candidates=demonstration_candidates,
                                                                       teacher_model=self)
            for item, prompt in zip(data_item, input_text_with_prompt):
                if self.config.get('data_name') != "KDD":
                    data_item_obj = {"prompt_text": prompt, "gold_label": item.label,
                                 "text": item.text,"query_str":item.query_str,"task_id":item.task_id,
                                 "query_lebal":item.query_lebal,"user_id":item.user_id,"query_id":item.query_id,
                                 "serp_id":item.serp_id,"order":item.order,"dwell_time":item.dwell_time,"his":item.his,"total_clicks_number":item.total_clicks_number,
                                 "clicked_ranks_list":item.clicked_ranks_list,"max_clicked_rank":item.max_clicked_rank,"avg_dwell_time":item.avg_dwell_time}
                else:
                    data_item_obj = {"prompt_text": prompt, "gold_label": item.label,
                                 "text": item.text, "query_str": item.query_str, "task_id": item.task_id,
                                 "query_lebal": item.query_lebal, "user_id": item.user_id, "query_id": item.query_id,
                                 "serp_id": item.serp_id, "order": item.order, "dwell_time": item.dwell_time,
                                 "his": item.his}

                writer_f.write(f"{json.dumps(data_item_obj,ensure_ascii=False)}\n")

        writer_f.close()
        return save_data_path

    def step2_get_gpt3_results(self, step1_prompt_data_path: str, resume: bool = False):
        thought_summary = False
        if self.config["data_name"] == "KDD":
            score_options = "1,2,3,4这四个打分"
        else:
            score_options = "0,1,2,3,4这五个打分"
        def format_document_new(item):
            his = item["his"] if item["his"] else "None"
            if len(item['clicked_ranks_list']) == 1:
                return (
                        item['text'] +
                        "。该文档是搜索者在该查询中第" + str(int(item['order']) + 1) + "个点击进入阅读的文档。" +
                        "该文档在文档列表中的排名是第" + str(int(item['serp_id']) + 1) + "名。" +
                        "搜索者只点击了这一个文档就完成了搜索。"+
                        "搜索者对该文档的阅读时间为" + str(np.ceil(item['dwell_time'] / 1000)) + "秒。"
                )
            else:
                return (
                        item['text'] +
                        "。该文档是搜索者在该查询中第" + str(int(item['order']) + 1) + "个点击进入阅读的文档。" +
                        "该文档在文档列表中的排名是第" + str(int(item['serp_id']) + 1) + "名。" +
                        "而搜索者在此查询下总共点击了" + str(int(item['total_clicks_number'])) + "个文档，" +
                        "点击的所有文档的排名是" + str([x + 1 for x in item['clicked_ranks_list']]) + "。" +
                        "最大的点击深度是" + str(item['max_clicked_rank'] + 1) + "。" +
                        "搜索者对该文档的阅读时间为" + str(np.ceil(item['dwell_time'] / 1000)) + "秒。" +
                        "而搜索者在此查询上的所有文档的平均阅读时长是" + str(np.ceil(item['avg_dwell_time'] / 1000)) + "秒。"
                )
        def query_level_feature(item, df):
            search_id = item["query_id"]
            row = df.iloc[search_id]
            assert int(item["query_lebal"]) == int(row['Q_SAT'])
            if row.empty:
                return f""
            session_end_text = "这是整个搜索会话的最后一个查询。" if row['isSessionEnd'] else "这不是整个搜索会话的最后一个查询。"
            prompt = (
                f"{item['text']}。该文档是搜索者第{int(item['order']) + 1}个点击进入阅读的文档，该文档在文档列表中的排名是第{int(item['serp_id']) + 1}名，"
                f"点击的文档的平均排名是 {round(row['AvgClickRank'], 1)}位。"
                f"点击的最深的文档是第 {int(round(row['ClickDepth']))}位。"
                f"信息搜索者对该文档的阅读时间为{np.ceil(item['dwell_time'] / 1000)}秒。"
                f"而搜索者在此查询中阅读每个文档的平均时间是 {round(row['AvgContent']/ 1000, 1)}秒，"
            )
            return prompt
        saved_result_path = os.path.join(self.config["save_log_dir"], "step2_result.jsonl")
        resume_offset = 0 if not resume else get_num_lines(saved_result_path)
        if os.path.exists(saved_result_path):
            resume=True
        if os.path.exists(saved_result_path) and not resume:
            raise FileExistsError(f"step2_get_gpt3_results -> {saved_result_path}")
        check_file_and_mkdir_for_save(saved_result_path, resume=resume, file_suffix=".jsonl")
        data_item_lst = load_jsonl(step1_prompt_data_path, offset=resume_offset)

        writer_mode = "w" if not resume else "a"
        writer_f = open(saved_result_path, writer_mode, encoding='utf-8')
        num_workers = int(self.config['num_agents'])
        if self.config["data_name"]=="KDD":
            file_path = './data/KDD19zong_query_id.xlsx'
            df = pd.read_excel(file_path)

        for idx, batch_data_item in tqdm(enumerate(chunked(data_item_lst, num_workers)),
                                         total=math.ceil(len(data_item_lst) / num_workers), desc="step-2"):
            batch_prompt_text_pre = [
                {"role": "system", "content": "你是信息检索领域的智能助手，能够根据信息搜索者的信息需求和查询对文档的有用性进行评分。"},
            ]
            batch_prompt_text= [copy.deepcopy(batch_prompt_text_pre) for _ in range(num_workers)]
            for i, item in enumerate(batch_data_item):
                print(i)
                batch_prompt_text[i].append({"role": "user",
                 "content": "现在的信息需求和查询为："+item["prompt_text"]+"。\n\n然后我会提供给你文档，你必须从" + score_options + "中选择一个代表此文档对于当前查询和信息需求的有用性程度，数字越大代表有用性越高。"})
                batch_prompt_text[i].append({'role': 'assistant', 'content': '好的，请提供文档。'})
                if self.config["data_name"]=="KDD":
                    batch_prompt_text[i].append({'role': 'user', 'content': query_level_feature(item, df)})
                else:
                    batch_prompt_text[i].append({'role': 'user', 'content': format_document_new(item)})
                batch_prompt_text[i].append({'role': 'assistant', 'content': '已接收该文档。'})
                if thought_summary==False:
                    batch_prompt_text[i].append({"role": "user",'content':"现在请你从" + score_options + "中选择一个代表该文档对于当前查询和信息需求的有用性打分，数字越大代表有用性越高。请先说出你的思考，然后再输出你打的分数，格式为：思考\n\n\n\n打分。"})
                else:
                    batch_prompt_text[i].append({"role": "user",'content':"现在请你从" + score_options + "中选择一个代表该文档对于当前查询和信息需求的有用性打分，数字越大代表有用性越高。提示：你可以参考“有帮助”、“详细”、“相关”、“百科”、“具体”、“全面”这六个方面或其中几个方面进行考量，请先说出你的思考，然后再输出你打的分数，格式为：思考\n\n\n\n打分。"})

            api_keys = list(self.config['api_keys'])
            num_agents = int(self.config['num_agents'])
            gpt_returned_results = self.model.forward(self.config["model_name"], api_keys, num_agents,batch_prompt_text,)
            gpt_returned_text = gpt_returned_results
            gpt_returned_logprobs = len(gpt_returned_results) * [None]
            for data_item, returned_text, returned_logprobs, batch_prompt_text_inputgpt in zip(batch_data_item, gpt_returned_text,
                                                                   gpt_returned_logprobs,batch_prompt_text):
                save_result_item = {"gpt_returned_result": returned_text, "gold_label": data_item["gold_label"], "prompt_text": str(batch_prompt_text_inputgpt),
                                     "text": data_item["text"],
                                    "query_str": data_item["query_str"], "task_id": data_item["task_id"],
                                    "query_lebal": data_item["query_lebal"], "user_id": data_item["user_id"], "query_id": data_item["query_id"],
                                    "serp_id": data_item["serp_id"], "order": data_item["order"],"dwell_time":data_item["dwell_time"]
                                    }
                writer_f.write(f"{json.dumps(save_result_item,ensure_ascii=False)}\n")
        writer_f.close()
        return saved_result_path

    def step3_map_competition_result_to_label(self, step2_gpt_result_path: str):
        """
        step_3
        """
        if self.config["data_name"] == "KDD":
            score_options = "1,2,3,4这四个打分"
        else:
            score_options = "0,1,2,3,4这五个打分"
        saved_result_path_lst = []
        feasible = True
        saved_result_path = os.path.join(self.config.save_log_dir, f"step3_result.jsonl")
        save_out_of_scope_result_path = os.path.join(self.config.save_log_dir,
                                                     f"step3_out_of_scope_prediction.jsonl")
        if os.path.exists(saved_result_path) or os.path.exists(save_out_of_scope_result_path):
            raise FileExistsError(f"step3_map_competition_result_to_label -> {saved_result_path}")
        saved_result_path_lst.append(saved_result_path)

        data_item_lst = load_jsonl(step2_gpt_result_path)
        out_of_scope_pred_lst = []
        update_data_item_lst = []
        error_data_item_lst=[]
        for data_item in data_item_lst:
            try:
                pred_label = data_item["gpt_returned_result"]
                pred_label_word_in_verbalizer = True
                data_item.update(
                    {"pred_label": pred_label, "pred_label_word_in_verbalizer": pred_label_word_in_verbalizer,
                     })
                update_data_item_lst.append(data_item)
            except LookupError:
                error_data_item_lst.append(data_item)
        num_workers = int(self.config['num_agents'])
        for idx, batch_data_item in tqdm(enumerate(chunked(error_data_item_lst, num_workers)),
                                         total=math.ceil(len(error_data_item_lst) / num_workers), desc="step-3"):
            batch_prompt_text = [
                [
                    {"role": "user", "content": "你是一个文档有用性打分器。你刚刚已经输出了以下内容："},
                    {"role": "assistant", "content": f'"{item["gpt_returned_result"]}"'},
                    {"role": "user", "content": "接下来请你总结你刚刚已输出的思考直接输出你的打分，请你只输出"+score_options+"中的一个数字代表打分，不要说任何解释和思考。"}
                ]
                for item in batch_data_item
            ]
            api_keys = list(self.config['api_keys'])
            num_agents = int(self.config['num_agents'])
            gpt_returned_results2 = self.model.forward(self.config["model_name"], api_keys, num_agents, batch_prompt_text,)
            for previous_item, gpt_returned_results2_str in zip(batch_data_item,gpt_returned_results2):
                previous_item["gpt_returned_result"]=previous_item["gpt_returned_result"]+"总结来说，"+gpt_returned_results2_str
                try:
                    pred_label = previous_item["gpt_returned_result"]
                    pred_label_word_in_verbalizer = True
                    previous_item.update(
                        {"pred_label": pred_label, "pred_label_word_in_verbalizer": pred_label_word_in_verbalizer,
                         })
                    update_data_item_lst.append(previous_item)
                except LookupError:
                    pass

        if feasible:
            save_jsonl(saved_result_path, update_data_item_lst)

            save_jsonl(save_out_of_scope_result_path, out_of_scope_pred_lst)
        return saved_result_path_lst


def run():
    import time

    start_time = time.time()
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", default=2333, type=int, help="random seed")
    parser.add_argument("--random", action="store_true", default=False)
    parser.add_argument("--config_path", default="../config/", type=str, help="path to the config file.")
    parser.add_argument("--test_file_name", default="test", type=str)
    parser.add_argument("--step_idx", default="", type=str)
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--fix_uid", default=False, action="store_true")
    parser.add_argument("-c", "--config_file", type=str, default='config/config.yaml', help="Path to config file")
    args = parser.parse_args()

    import os
    import json



    step1_save_path, step2_save_path, step3_save_path, = None, None, None
    step_idx_lst = [int(idx) for idx in args.step_idx.split("-")]
    config = CfgNode(new_allowed=True)
    config.merge_from_file(args.config_file)


    gpt3_for_text_cls_task = GPT3TextCLS(config=config)

    if 1 in step_idx_lst:
        resume = False
        import os

        file_path = r"config/step1_data.jsonl"
        if os.path.exists(file_path):
            print(os.path.exists(file_path))
            resume = True

        step1_save_path = gpt3_for_text_cls_task.step1_prepare_input(
                                                                     resume=resume)
        print(step1_save_path)
    if 2 in step_idx_lst:
        if step1_save_path is None:
            step1_save_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir, "step1_data.jsonl")
        step2_save_path = gpt3_for_text_cls_task.step2_get_gpt3_results(step1_save_path,
                                                                        resume=True if args.resume and step_idx_lst[
                                                                            0] == 2 else False)
        print(step2_save_path)
    if 3 in step_idx_lst:
        saved_result_path = os.path.join(gpt3_for_text_cls_task.config["save_log_dir"], "step2_result.jsonl")
        step3_save_path = gpt3_for_text_cls_task.step3_map_competition_result_to_label(saved_result_path)

if __name__ == "__main__":
    run()