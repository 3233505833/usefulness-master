
from utils.new_data_loader import AbsDataloader
import json
import os
from typing import Dict, List, Union
from utils.new_data_retriever import SimCSERetriever
from utils.new_data_utils import Detokenizer,  encode_md5hash

from utils.new_file_utils import  DataItem
class Prompt(object):


    __slots__ = ["model_backbone", "prompt_strategy", "instance_num", "instance_strategy", "gradient_update"]

    def __init__(self, key_value_params: Dict = None):
        if key_value_params is not None:
            for key in self.__slots__:
                if key in key_value_params.keys():
                    self.__setattr__(key, key_value_params[key])
                    key_value_params.pop(key)
                else:
                    self.__setattr__(key, None)
            if len(key_value_params) != 0:
                raise ValueError(key_value_params)



    def map_predicted_verbalizer_to_label(self):
        raise NotImplementedError

    def _get_config(self):
        config_pairs = {}
        for slot_key in self.__slots__:
            if slot_key in ["tokenizer", "data_retriever", "detokenizer"]:
                slot_value = None
            elif slot_key in ["dataloader"]:
                slot_value = str(self.__getattribute__(slot_key))
            else:
                try:
                    slot_value = self.__getattribute__(slot_key)
                except:
                    slot_value = None
            config_pairs[slot_key] = slot_value
        return config_pairs

    def __str__(self):
        """return the string."""
        config_data = self._get_config()
        return json.dumps(config_data, indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, config_path: str):
        """load config from json assets."""
        with open(config_path, "r", encoding="utf-8") as f:
            config_items = json.load(f)
        filtered_configs = {key: value for key, value in config_items.items() if key in cls.__slots__}
        return cls(filtered_configs)

    def save_to_json(self, save_path: str):
        """save config to file."""
        config_pairs = self._get_config()
        if os.path.exists(save_path):
            raise FileExistsError(f"{save_path}")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_pairs, f, sort_keys=True, indent=2, ensure_ascii=False)
        print(f"SAVE CONFIG TO {save_path}")


class GPT3FewShotSamplingPrompt(Prompt):
    __slots__ = Prompt.__slots__ + ["task_description", "delimiter", "demonstration_pattern", "verbalizer",
                                    "feasible_verbalizer",
                                    "assemble_demonstration_strategy", "max_prompt_len", "inverse_verbalizer",
                                    "detokenizer", "verbalizer_position_idx", "demonstration_subtask_description",
                                    "assemble_demonstration_pattern", "data_retriever", "data_retriever_candidate_dir",
                                    "retriever_name_or_path",
                                    "retriever_ckpt_path", "file_saved_retriever_results", "demonstration_ranking",
                                    "non_verbalizer", "dataloader", "max_instance_len", "max_explain_len",
                                    "model_generate_max_len", "demonstration_subtask_description_pos", "prompt_suffix","background"]

    def __init__(self, key_value_params: Dict = None):
        super(GPT3FewShotSamplingPrompt, self).__init__(key_value_params)
        import yaml
        config_path = './config/config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        if config.get('data_name')=="KDD":
            self.background = {
            }
        else:
            self.background = {
            }

        self.dataloader = AbsDataloader()
        self.instance_num = 3
        self.max_instance_len = 200
        self.assemble_demonstration_strategy = "model_generate"

        self.max_explain_len = 100
        self.instance_strategy = "no-simcse-nearest-neighbor"
        self.demonstration_subtask_description_pos = 0
        self.delimiter="\n\n"
        self.max_prompt_len = 1024
        self.demonstration_subtask_description="首先，请先提供该文档能够满足查询的对应线索。例如文档中的某些部分是不是回答了查询的问题或者满足了信息需求。请分条以以下的格式列出：<查询或信息需求>:\"获取的信息\"。\"。然后根据列出的对应线索，请你给出支持该文档被搜索引擎用户判定为该有用性等级的推理。"
        self.verbalizer = {
            "4": "非常有用",
            "3": "有用",
            "2": "有一点用",
            "1": "没用",
            "0": "完全没用",
        }
        self.feasible_verbalizer = {label_id: label_token.lower() for label_id, label_token in
                                        self.verbalizer.items()}
        self.inverse_verbalizer = {}
        for label_symbol, label_word in self.feasible_verbalizer.items():
            if isinstance(label_word, list):
                for token in label_word:
                    assert not any(element.isupper() for element in token)
                    self.inverse_verbalizer[token] = label_symbol
            elif isinstance(label_word, str):
                assert not any(element.isupper() for element in label_word)
                self.inverse_verbalizer[label_word] = label_symbol
            else:
                raise ValueError(self.inverse_verbalizer)
        self.detokenizer = Detokenizer()

        if self.instance_strategy == "simcse-nearest-neighbor":
            data_retriever_loader = self.dataloader

            self.data_retriever = SimCSERetriever(mlm_name_or_path="../data/models/bert-base-chinese", max_len=256,
                                                  saved_nearest_neighbor_file=None)
            self.data_retriever.build_index(data_retriever_loader)


    def select_demonstration_instances(self, demonstrations_candidates: List[DataItem] = None,
                                       test_instance: Union[List[str,], str] = None,
                                       shuffle: bool = True, ) -> List[DataItem]:
        sampled_demonstration_lst = self.data_retriever.search(test_instance, top_k=self.instance_num)
        sampled_demonstration_text_lst = [item[0] for item in sampled_demonstration_lst]
        sampled_demonstration_label_lst = [self.data_retriever.text_md5_to_label[encode_md5hash(text_item)] for
                                   text_item in sampled_demonstration_text_lst]#[1]
        sampled_demonstration_query_str_lst=[self.data_retriever.text_md5_to_query_str[encode_md5hash(text_item)] for
                                   text_item in sampled_demonstration_text_lst]#[2]

        sampled_demonstration_task_id_lst = [self.data_retriever.text_md5_to_task_id[encode_md5hash(text_item)] for
                                       text_item in sampled_demonstration_text_lst]  # [3]

        sampled_demonstration_lst = []
        for text_item, label_item,query_str_item,task_id_item in zip(sampled_demonstration_text_lst, sampled_demonstration_label_lst, sampled_demonstration_query_str_lst,sampled_demonstration_task_id_lst):
            if text_item =="" or len(text_item)<60:
                continue
            sampled_demonstration_lst.append(DataItem(text=text_item, label=label_item,query_str=query_str_item,task_id=task_id_item))
        sampled_demonstration_lst = sampled_demonstration_lst
        return sampled_demonstration_lst

    def assemble_demonstrations_batch(self, sampled_demonstration_lst_batch: List[List[DataItem]] = None,
                                      teacher_model=None,
                                      max_len: int = 2048, ) -> str:

        demonstration_info_batch = []
        demonstration_prompt_batch = []
        sampled_demonstration_obs_batch=[]
        for sampled_demonstration_lst in sampled_demonstration_lst_batch:
            sampled_demonstration_obs = [item.text for item in
                                          sampled_demonstration_lst]
            sampled_demonstration_obs_batch.append(sampled_demonstration_obs)

            self.demonstration_pattern = "信息需求为： <BACKGROUND> \n查询为： <QUERY> \n文档的主要内容为: <TEXT>\n文档有用性: <VERBALIZER-LABEL>"


            demonstration_prompt_subtext = []
            for item, obs in zip(sampled_demonstration_lst,sampled_demonstration_obs):
                replaced_prompt = self.demonstration_pattern.replace("<TEXT>", obs)
                demonstration_prompt_subtext.append(replaced_prompt)

            demonstration_prompt_subtext = [
                item.replace("<VERBALIZER-LABEL>", self.verbalizer[str(sampled_demonstration_lst[idx].label)]) for
                idx, item in
                enumerate(demonstration_prompt_subtext)]
            demonstration_prompt_subtext = [item.replace("<BACKGROUND>", str(
                self.background[sampled_demonstration_lst[idx].task_id]['background'] +
                self.background[sampled_demonstration_lst[idx].task_id]['goal'])) for idx, item in
                                            enumerate(demonstration_prompt_subtext)]
            demonstration_prompt_subtext = [item.replace("<QUERY>", str(sampled_demonstration_lst[idx].query_str)) for
                                            idx, item in
                                            enumerate(demonstration_prompt_subtext)]
            if self.demonstration_subtask_description_pos == 0:
                demonstration_prompt = [self.demonstration_subtask_description + f"{self.delimiter}" + item for item
                                        in
                                        demonstration_prompt_subtext]
            elif self.demonstration_subtask_description_pos == -1:
                demonstration_prompt = [item + f"{self.delimiter}" + self.demonstration_subtask_description for item
                                        in
                                        demonstration_prompt_subtext]
            else:
                raise ValueError(self.demonstration_subtask_description_pos)
            demonstration_prompt_batch.append(demonstration_prompt)
        transposed_list = list(zip(*demonstration_prompt_batch))
        demonstration_prompt_batch = [list(row) for row in zip(*transposed_list)]
        model_generated_info_batch=[]
        for i in range(len(demonstration_prompt_batch)):
            self.model_generate_max_len = 3000
            model_generated_info = teacher_model.forward(demonstration_prompt_batch[i])

            assert len(model_generated_info) == len(demonstration_prompt_batch[i])
            model_generated_info_batch.append(model_generated_info)
        transposed_generated_list = list(zip(*model_generated_info_batch))
        model_generated_info_batch = [list(row) for row in zip(*transposed_generated_list)]
        for model_generated_info, sampled_demonstration_lst, sampled_demonstration_obs in zip(model_generated_info_batch,
                                                                   sampled_demonstration_lst_batch,sampled_demonstration_obs_batch):
            demonstration_info = ""
            model_generated_info = [item.strip().replace("\n\n", "\n") for item in model_generated_info]
            model_generated_info = [self._clip_text_by_space_len(item, self.max_explain_len) for item in
                                    model_generated_info]
            assert len(model_generated_info) == len(sampled_demonstration_lst)
            self.assemble_demonstration_pattern = "信息需求为： <BACKGROUND> \n查询为： <QUERY> \n文档的全部内容为： <TEXT> \n对应线索和推理过程: <MODEL-GENERATE>\n文档有用性 <VERBALIZER-LABEL>"
            for demon, model_gen, obs in zip(sampled_demonstration_lst, model_generated_info,sampled_demonstration_obs):
                current_demon_info = self.assemble_demonstration_pattern.replace("<TEXT>",
                                                                                     obs,
                                                                                     )
                current_demon_info = current_demon_info.replace("<BACKGROUND>",
                                                                self.background[demon.task_id]['background'] +
                                                                self.background[demon.task_id]['goal'])
                current_demon_info = current_demon_info.replace("<QUERY>", demon.query_str)
                current_demon_info = current_demon_info.replace("<VERBALIZER-LABEL>",
                                                                self.verbalizer[str(demon.label)])
                current_demon_info = current_demon_info.replace("<MODEL-GENERATE>", model_gen)
                demonstration_info += current_demon_info + self.delimiter
            demonstration_info_batch.append(demonstration_info)

        return demonstration_info_batch


    def get_model_input_batch(self, instance_text_batch: List[str], data_item: List[DataItem],demonstrations_candidates: List[DataItem] = None,
                              sampled_demonstration_lst: List[DataItem] = None, teacher_model=None,
                              need_detokenize: bool = True, max_len: int = None) -> List[str]:
        user_few=False
        self.task_description = "你是一个文档有用性打分器。请按严格按照以下格式写出三个部分的内容：\n\n1. 文档和信息需求之间的对应线索\n2. 文档是否有用的推理过程\n3. 文档有用性打分\n\n要求如下：\n\n在“文档和信息需求之间的对应线索”部分，请分条以以下的格式列出：<查询或信息需求的原文>：“截取的文档原文”，以下是一个例子：“信息需求为：\n请根据列举一些国际知名的电子产品制造商和它们的产品范围，请根据搜索结果，介绍两个国际知名的电子产品制造商及其主要产品范围，以及创新成就。\n查询为：国际知名的电子产品制造商\n文档的全部内容为：\n联想公司是一家知名的电脑制造公司。该公司的主要工作领域包括电脑硬件制造、软件开发和信息技术服务。联想在电脑行业取得了许多重大贡献，其中包括推出了多款具有领先技术的电脑产品，如ThinkPad系列笔记本电脑和Yoga系列二合一笔记本电脑。\n你应该列出以下线索：<电子产品制造商>:联想公司是一家知名的电脑制造公司。\n<产品范围>:该公司的主要工作领域包括电脑硬件制造、软件开发和信息技术服务。\n<创新成就>:联想在电脑行业取得了许多重大贡献，其中包括推出了多款具有领先技术的电脑产品，如ThinkPad系列笔记本电脑和Yoga系列二合一笔记本电脑。”\n\n在“文档是否有用的推理过程”部分，请结合对应线索部分，参考“有帮助”、“详细”、“相关”、“百科”、“具体”、“全面”这六个方面或其中几个，写出你对文档的思考。\n\n在“文档有用性打分”部分，请结合前面找出的对应线索和推理过程，给出文档在实现该信息需求时是否有用，请从“非常有用“,“有用“,“有一点用“,“没用“,“完全没用“中选择一个，格式为“文档有用性打分：你的选择”，不要输出自己的思考以及其他任何额外的信息。",
        if user_few==True:
            if sampled_demonstration_lst is None:
                sampled_demonstration_batch_lst = [self.select_demonstration_instances(demonstrations_candidates,
                                                                                       test_instance=instance_text) for
                                                   instance_text in instance_text_batch]

            demonstration_info_batch = self.assemble_demonstrations_batch(sampled_demonstration_batch_lst,
                                                                          teacher_model=teacher_model,
                                                                          max_len=self.max_prompt_len - 450)

            if need_detokenize:
                instance_text_batch = [self.detokenizer.detokenize(instance_text) for instance_text in instance_text_batch]
            instance_text_batch = [self._clip_text_by_space_len(instance_text, self.max_instance_len) for instance_text in
                                   instance_text_batch]

            model_input_instance_batch = [
                f"以下是几个文档打分的例子：{demonstration_info}以下是你需要打分的文档: 信息需求为： <BACKGROUND> \n查询为： <QUERY> \n" for
                instance, demonstration_info
                in zip(instance_text_batch, demonstration_info_batch)]

        else:
            if isinstance(self.task_description, tuple):
                self.task_description = self.task_description[0]
            model_input_instance_batch = [
                f"信息需求为： <BACKGROUND> \n查询为： <QUERY> \n" for
                instance
                in instance_text_batch]


        max_len = self.max_prompt_len if max_len is None else max_len
        if self.max_prompt_len <= max(
                [len(model_input_instance.split(" ")) for model_input_instance in model_input_instance_batch]):
            print(f"WARNING: PROMPT IS TOO LONG.")
            model_input_instance_batch = [self._clip_text_by_space_len(model_input, max_len) for model_input in
                                          model_input_instance_batch]
        model_input_instance_batch_new=[]
        for demon, current_demon_info in zip(data_item, model_input_instance_batch):
            current_demon_info = current_demon_info.replace("<BACKGROUND>",
                                                            self.background[demon.task_id]['background'] +
                                                            self.background[demon.task_id]['goal'])
            current_demon_info = current_demon_info.replace("<QUERY>", demon.query_str)
            model_input_instance_batch_new.append(current_demon_info)
        return model_input_instance_batch_new

    def _clip_text_by_space_len(self, input_text: str, max_space_len: int = 200) -> str:
        input_token = input_text.split(" ")
        if len(input_token) <= max_space_len:
            return input_text

        input_token_clipped = input_token[:max_space_len]
        input_text_clip = " ".join(input_token_clipped)
        return input_text_clip




