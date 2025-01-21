
import hashlib
import os
import re
from collections import namedtuple
from typing import Dict, List
from transformers import AutoTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Tokenizer(object):
    def __init__(self, llm_dir: str, do_lower_case: bool = False, max_len: int = 512, pad_to_max_length: bool = False,
                 add_special_tokens: bool = True, return_offsets_mapping: bool = True):
        self.llm_dir = llm_dir
        self.do_lower_case = do_lower_case
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_dir,
                                                       do_lower_case=self.do_lower_case,
                                                       use_fast=True,
                                                       do_basic_tokenize=False)
        self.pad_to_max_length = pad_to_max_length
        self.add_special_tokens = add_special_tokens
        self.return_offsets_mapping = return_offsets_mapping

    def __len__(self) -> int:
        return len(self._get_vocab_idx2token())

    def _get_vocab_idx2token(self) -> dict:
        return {value: key for key, value in self.tokenizer.vocab.items()}

    def _clip_to_maxlen(self, input_batch: List[str]) -> List[str]:
        clipped_input_tokens_batch = [input_item.split(" ")[: self.max_len] for input_item in input_batch]
        clipped_input_batch = [" ".join(clipped_item) for clipped_item in clipped_input_tokens_batch]
        return clipped_input_batch

    def decode(self, idx_batch: List[List[int]]) -> List[str]:
        vocab_idx2token = self._get_vocab_idx2token()
        str_token_batch = [[vocab_idx2token[item] for item in idx_item] for idx_item in idx_batch]
        text_str = [" ".join(str_token_item) for str_token_item in str_token_batch]
        return text_str

    def tokenize_input_batch(self, input_batch: List[str], ) -> Dict:
        cliped_input_batch = self._clip_to_maxlen(input_batch)
        tokenizer_output = self.tokenizer.batch_encode_plus(cliped_input_batch,
                                                            pad_to_max_length=self.pad_to_max_length,
                                                            return_offsets_mapping=self.return_offsets_mapping,
                                                            add_special_tokens=self.add_special_tokens,
                                                            return_tensors="pt",
                                                            max_length=self.max_len
                                                            )

        return tokenizer_output

    def __str__(self):
        return "AutoTokenizer"




def encode_md5hash(input_str: str) -> str:
    """
    Desc:
        get the md5 value of the input string.
    Param:
        input_str:
    Return:
        md5_value(string) of the input string.
    """
    encode_result = hashlib.md5(input_str.encode())
    encode_md5value = encode_result.hexdigest()
    return encode_md5value



class Detokenizer(object):
    def __init__(self):
        self.detokenizer = TreebankWordDetokenizer()

    def detokenize(self, input_text: str, token_delimiter: str = " ") -> str:
        """
        Desc:
            Untokenizing a text undoes the tokenizing operation, restoring
            punctuation and spaces to the places that people expect them to be.
            Ideally, `untokenize(tokenize(text))` should be identical to `text`,
            except for line breaks.
            credit: https://stackoverflow.com/questions/21948019/python-untokenize-a-sentence
        """
        input_token_lst = input_text.split(token_delimiter)
        detokenized_text = self.detokenizer.detokenize(input_token_lst)
        detokenized_text = detokenized_text.strip()
        detokenized_text = detokenized_text.replace("-lrb-", "(")
        detokenized_text = detokenized_text.replace("-rrb-", ")")
        detokenized_text = detokenized_text.replace("`` ", '" ').replace(" ''", ' "').replace(". . .", "...")
        detokenized_text = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", detokenized_text)
        detokenized_text = re.sub(r" ([.,:;?!%]+)$", r"\1", detokenized_text)
        return detokenized_text.strip()

    def __str__(self):
        return "TreebankWordDetokenizer"
