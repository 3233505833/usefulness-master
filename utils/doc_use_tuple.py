coding = 'utf8'

import re
import json

def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"', }
    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        entity = sz.group()
        key = sz.group('name')
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:

            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr


def filter_tags(htmlstr):
    re_nav = re.compile('<nav.+</nav>')
    re_cdata = re.compile('//<!\[CDATA\[.*//\]\]>', re.DOTALL)
    re_script = re.compile(
        '<\s*script[^>]*>.*?<\s*/\s*script\s*>', re.DOTALL | re.I)
    re_style = re.compile(
        '<\s*style[^>]*>.*?<\s*/\s*style\s*>', re.DOTALL | re.I)
    re_textarea = re.compile(
        '<\s*textarea[^>]*>.*?<\s*/\s*textarea\s*>', re.DOTALL | re.I)
    re_br = re.compile('<br\s*?/?>')
    re_h = re.compile('</?\w+.*?>', re.DOTALL)
    re_comment = re.compile('<!--.*?-->', re.DOTALL)
    re_space = re.compile(' +')
    s = re_cdata.sub('', htmlstr)
    s = re_nav.sub('', s)
    s = re_script.sub('', s)
    s = re_style.sub('', s)
    s = re_textarea.sub('', s)
    s = re_br.sub('', s)
    s = re_h.sub('', s)
    s = re_comment.sub('', s)
    s = re.sub('\\t', '', s)
    s = re.sub(' ', '', s)
    s = re.sub('\\n', '', s)
    s = re_space.sub(' ', s)
    s = replaceCharEntity(s)
    return s
def get_doc_use_tuple(task_list):#UUST
    with open("./data/data_full_withSERPid.json", 'r', encoding='utf-8') as file:
        data_new = json.load(file)
    tuple_list = []
    for user in data_new:
        for task_id in user["task"]:
            if task_id in task_list:
                task_details = user["task"][task_id]
                for content_details in task_details["content"]:
                    query_id = str(content_details["query_id"])
                    serp_id = str(content_details["SERP_id"])
                    tuple_list_query=[]
                    no_screenshot=False
                    total_clicks_number = -1
                    clicked_ranks_list = []
                    max_clicked_rank = -1
                    avg_dwell_time_list = []
                    for rank in range(len(content_details["SERP"])):
                        if content_details["SERP"][rank]["click_or_not"] == 1:
                            try:
                                clicked_ranks_list.append(rank)
                                avg_dwell_time_list.append(int(content_details["SERP"][rank]["dwell_time"]))
                            except:
                                pass

                    total_clicks_number=len(clicked_ranks_list)
                    try:
                        max_clicked_rank=max(clicked_ranks_list)
                        avg_dwell_time = sum(avg_dwell_time_list) / len(avg_dwell_time_list)
                        avg_dwell_time = round(avg_dwell_time, 1)
                    except:
                        pass

                    for rank in range(len(content_details["SERP"])):
                        if content_details["SERP"][rank]["click_or_not"] == 1:
                            try:
                                with open("data\OCRtxtzhongnew\OCRtxtzhongnew\\"+serp_id+"\\"+str(rank)+".txt", 'r', encoding='utf-8') as file:
                                    content = file.read()
                                text=content[:8000]
                            except:
                                no_screenshot=True
                                try:
                                    text = filter_tags(content_details["SERP"][rank]["html"]).strip()
                                    if text=="":
                                        text =" "
                                except:
                                    text = " "
                            try:

                                filtered_text = text.replace('{', '').replace('}', '')
                                filtered_text = filtered_text.replace('[', '').replace(']', '')
                                titleed=content_details["SERP"][rank]["title"].replace('[', '').replace(']', '').replace('{',
                                                                                                                 '').replace(
                                    '}', '')
                                snippeted=content_details["SERP"][rank]["snippet"].replace('[', '').replace(']',
                                                                                                                '').replace(
                                    '{', '').replace('}', '')
                                tuple_list_query.append(
                                    tuple((
                                        titleed + snippeted + filtered_text,
                                        total_clicks_number,
                                        clicked_ranks_list,
                                        max_clicked_rank,
                                        avg_dwell_time,
                                        content_details["SERP"][rank]["usefulness"],
                                        str(content_details["query"]),
                                        str(task_id),
                                        content_details["satisfaction"],
                                        user["user_id"],
                                        query_id,
                                        rank,
                                        int(content_details["SERP"][rank]["order"]),
                                        int(content_details["SERP"][rank]["dwell_time"]),
                                        no_screenshot,
                                        titleed,
                                        snippeted
                                    ))
                                )
                            except:
                                pass

                    sorted_tuple_list_query = sorted(tuple_list_query, key=lambda x: x[-5])
                    his_now=""
                    for item in sorted_tuple_list_query:

                        tuple_list.append(item[:-3]+(his_now,))

                        his_now+="\n"
                        his_now+=item[-2]+item[-1]

                    print(len(tuple_list[0]))
    return tuple_list


def get_doc_use_tuple_from_KDD_json(task_list):

    with open("data/KDD_v7.json", 'r', encoding='utf-8') as file:
        data_new = json.load(file)

    tuple_list = []

    for query_key in data_new:
        task_details = data_new[query_key]

        if str(task_details["task_id"]) in task_list:
            user_id = task_details["studentID"]
            feature_id = task_details["feature_id"]

            tuple_list_query = []
            no_screenshot = False

            for action in task_details["action_clicks"]:
                no_screenshot = True
                try:
                    content = filter_tags(action["content"]).strip()
                except:
                    content = " "

                filtered_text = content.replace('{', '').replace('}', '').replace('[', '').replace(']', '')
                title = ""
                snippet = filtered_text.split("推荐您")[0]

                try:
                    a=action["usefulness_score"]
                    tuple_list_query.append((
                        filtered_text,

                        action["usefulness_score"],
                        query_key.split("_")[0],
                        str(task_details["task_id"]),
                        str(task_details["query_sat"]),
                        user_id,
                        feature_id,
                        action["content_rank"].split("_")[1],
                        int(action["click_time"]),
                        int(action["time_diff"])*1000,
                        no_screenshot,
                        title,
                        snippet
                    ))
                except:
                    pass

            sorted_tuple_list_query = sorted(tuple_list_query, key=lambda x: x[-5])
            his_now = ""
            order=-1
            for item in sorted_tuple_list_query:
                order+=1
                item_list = list(item)
                item_list[8] = order
                item=tuple(item_list)
                tuple_list.append(item[:-3] + (his_now,))
                his_now += "\n"
                his_now += item[-1]

    return tuple_list
