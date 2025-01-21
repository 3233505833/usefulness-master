import json
from collections import defaultdict
def group_5_rater(judge_pre_file, judge_file, stage):
    label_sums = defaultdict(lambda: {'sum': 0, 'count': 0})
    with open(judge_pre_file, 'r',encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            prompt = data['prompt_text']+str(data['query_id'])
            label_sums[prompt]['sum'] += int(data['pred_label'])
            label_sums[prompt]['count'] += 1
            label_sums[prompt]["text"] = data["text"]
            label_sums[prompt]["query_str"] = data["query_str"]
            label_sums[prompt]["task_id"] = data["task_id"]
            label_sums[prompt]["query_lebal"] = data["query_lebal"]
            label_sums[prompt]["user_id"] = data["user_id"]
            label_sums[prompt]["query_id"] = data["query_id"]
            label_sums[prompt]["serp_id"] = data["serp_id"]
            label_sums[prompt]["order"] = data["order"]
            label_sums[prompt]["gold_label"] = data["gold_label"]
            label_sums[prompt]["his"] = data["his"]
            label_sums[prompt]["dwell_time"] = data["dwell_time"]

            if 'pred_label_list' not in label_sums[prompt]:
                label_sums[prompt]['pred_label_list'] = []
                label_sums[prompt]['pred_label_list'].append(int(data['pred_label']))
            else:
                label_sums[prompt]['pred_label_list'].append(int(data['pred_label']))

    writer_f = open(judge_file, 'w', encoding='utf-8')
    for prompt, data in label_sums.items():
        if len(label_sums[prompt]['pred_label_list']) > 3 or len(label_sums[prompt]['pred_label_list']) == 3:
            result = {
                'prompt_text': prompt,
                'sum': data['sum'],
                'count': data['count'],
                "pred_label": stage,
                "text": data["text"],
                "query_str": data["query_str"],
                "task_id": data["task_id"],
                "query_lebal": data["query_lebal"],
                "user_id": data["user_id"],
                "query_id": data["query_id"],
                "serp_id": data["serp_id"],
                "order": data["order"],
                "gold_label": data["gold_label"],
                "his":data["his"],
                "dwell_time": data["dwell_time"]
            }
            writer_f.write(f"{json.dumps(result, ensure_ascii=False)}\n")
    writer_f.close()
