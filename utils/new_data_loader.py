import csv
import os.path
import random
from collections import Counter
from typing import List

from tqdm import tqdm
from collections import namedtuple


from utils.new_file_utils import load_jsonl, DataItem

class AbsDataloader(object):
    """Abstract class for dataset."""

    def __init__(self):
        pass

    def count_label_dist(self, ):
        raise NotImplementedError

    def load_data_files(self) -> List[DataItem]:
        data_file = ""
        data_items = self.load_tsv_file(data_file)
        return data_items

    @classmethod
    def load_tsv_file(cls, data_file_path: str, delimiter: str = "\t", skip_header: bool = False):
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"FILE {data_file_path} NOT EXISTS ...")

        with open(data_file_path, "r", encoding="utf-8") as f:
            data_items = [tuple(item.replace("\n", "").split(delimiter)) for item in f.readlines()]
            data_items = [DataItem(text=item[0], label=item[1]) for item in data_items]
        if skip_header:
            data_items = data_items[1:]
        return data_items

    @classmethod
    def get_labels(cls):
        raise NotImplementedError

    @classmethod
    def load_csv_file(cls, data_file_path: str, skip_header: bool = False, delimiter: str = ",",
                      concat_text: bool = True):
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"FILE {data_file_path} NOT EXISTS ...")
        data_items = []
        with open(data_file_path, "r", newline='') as csvf:
            file_reader = csv.reader(csvf, delimiter=delimiter, )
            for item in file_reader:
                if not concat_text:
                    text = [item[1], item[2]]
                else:
                    text = f"{item[1]} {item[2]}"
                data_items.append(DataItem(text=text, label=item[0], title=item[1], desc=item[2]))

                if skip_header:
                    data_items = data_items[1:]
        return data_items

    @classmethod
    def load_jsonl_file(cls, load_file_path: str, text_key: str = "text", label_key: str = "label"):
        data_items = load_jsonl(load_file_path)
        data_items = [DataItem(text=item[text_key], label=item[label_key]) for item in data_items]
        return data_items


