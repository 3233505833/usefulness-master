
import csv
import json
import os.path
from typing import List, Dict

from tqdm import tqdm

from collections import namedtuple
DataItem = namedtuple("DataItem", ["text", "label", "query_str", "task_id","query_lebal","user_id","query_id","serp_id","order","dwell_time","his"], defaults=(None, None, None, None,  None, None,None, None,None,None))
DataItem_new = namedtuple("DataItem_new", ["text", "total_clicks_number","clicked_ranks_list","max_clicked_rank","avg_dwell_time","label", "query_str", "task_id","query_lebal","user_id","query_id","serp_id","order","dwell_time","his"], defaults=(None, None, None, None,  None, None,None, None,None,None))

def get_subdir_names(dir_path: str) -> List[str]:
    """Get a list of immediate subdirectories"""
    return next(os.walk(dir_path))[1]

def save_jsonl(save_file_path: str, data_item_lst: List, resume: bool = False):
    check_file_and_mkdir_for_save(save_file_path, resume=resume, file_suffix=".jsonl")
    mode = "w" if not resume else "a"
    with open(save_file_path, mode, encoding='utf-8') as f:
        for data_item in tqdm(data_item_lst, total=len(data_item_lst)):
            if isinstance(data_item, dict):
                json.dump(data_item, f, ensure_ascii=False)
                f.write('\n')
            elif isinstance(data_item, DataItem):
                json.dump({'text': data_item.text, 'label': data_item.label}, f, ensure_ascii=False)
                f.write('\n')
            else:
                raise ValueError

    print(f"INFO: SAVE {save_file_path}")

def save_json(save_file_path: str, data_item: Dict, resume: bool = False):
    check_file_and_mkdir_for_save(save_file_path, resume=resume, file_suffix=".json")
    with open(save_file_path, "w", encoding='utf-8') as f:
        json.dump(data_item, f, default=lambda o: o.__dict__, indent=4, ensure_ascii=False)
    print(f"INFO: save to {save_file_path}")


def check_file_and_mkdir_for_save(save_file_path: str, resume: bool = False, file_suffix: str = "jsonl"):
    assert save_file_path.endswith(f"{file_suffix}")
    if os.path.exists(save_file_path) and not resume:
        raise ValueError(f"{save_file_path} already exists ... ...")

    output_dir = os.path.dirname(save_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"INFO: Make directory {output_dir}")



def load_jsonl(load_file_path: str, offset: int = 0) -> List:
    assert load_file_path.endswith(".jsonl")

    if not os.path.exists(load_file_path):
        raise ValueError(f"{load_file_path} NOT exists ... ...")

    with open(load_file_path, "r", encoding="utf-8") as f:

        data_item_lst = []
        datalines = f.readlines()
        for data_item in tqdm(datalines, total=len(datalines)):
            data_item_lst.append(json.loads(data_item.strip()))

        assert offset >= 0 and offset <= len(data_item_lst)
        if offset != 0:
            data_item_lst = data_item_lst[offset:]
        return data_item_lst

def get_num_lines(file_path: str, strip_line: bool = True) -> int:
    if not os.path.exists(file_path):
        raise ValueError(f"{file_path} NOT exists ... ...")

    with open(file_path, "r", encoding= "utf-8") as f:
        datal = f.readlines()

    if strip_line:
        datal = [item.strip() for item in datal]
        datal_filtered = [item for item in datal if len(item) != 0]
        return len(datal_filtered)

    return len(datal)
