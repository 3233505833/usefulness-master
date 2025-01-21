# GPT
The framework of this project is based on a text classification code project.

For `pointwise.py`, `listwise.py`, and `op_ex_pairwise.py`, simply right-click to run our code. In the `.py ` code, modify the flag variables to include: whether to load unfinished `.jsonl `files and continue prediction, whether to add guidelines summarized from thoughts, and whether to use GPT-3 or GPT-4, which can be changed in the config file.

# LLAMA

## Step 0：Prepare Data

First, place the training and test datasets in the `zhdata` directory, and set the validation split ratio from the training set in `my.yml`.

## Step 1: Fine-Tuning

```
python trl_finetune.py -c configs/my.yml
```

## Step 2: Merge Model

```
python merge_lora.py -c configs/my.yml
```

## Step 3: Run Inference

```
python inference.py
```



