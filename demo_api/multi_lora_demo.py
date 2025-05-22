import gevent
from gevent import monkey
monkey.patch_all()
import re
import argparse

import gradio as gr
import mdtex2html
import zhconv

from multi_lora_model import load_predictors, Scenes
from classify import get_classifier
from pluginer import search


plugin_pattern = re.compile(r"地址实体:(?P<name>.*?),")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/home/xy/.paddlenlp/models/THUDM/chatglm-6b", help="The directory of model.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=2048, help="The batch size of data.")
    parser.add_argument("--tgt_length", type=int, default=160, help="The batch size of data.")
    parser.add_argument("--lora_dir", default="./models/scenes", help="The directory of LoRA parameters")
    parser.add_argument("--scenes_name", type=list, default=["hospital", "school", "district", "transportation", "community"], help="chosen scenes")
    return parser.parse_args()


args = parse_arguments()
predictor = load_predictors(args)
print(f"predictors={predictor}")


def angle2half(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def input_preprocess(line):
    line = ''.join([angle2half(item) for item in list(line.strip())])
    line = zhconv.convert(line, 'zh-ch')
    return line


case = ["bot: 请问您要去哪里?\n\nuser: 我要去九三花社。", "bot: 请问您要去哪里？\n\nuser: 去九号小区。", "bot: 请问您要去哪里？\n\nuser: 去花椒树社区。", "bot: 请问您要去哪里？\n\nuser: 去秦晋家园。\n\nbot: "]
all_texts = [
    "bot: 请问您要去哪里?\n\nuser:风雅园\n\nbot: 前文user回复中有地址实体:风雅园, 需要调知识库获取相关信息。\n\n知识库: [{'名称': '风雅园', '距离': '5公里外', '区划': '昌平区', '子节点': {'出入口': [{'名称': '西门', '地址': '风雅园二区1号正西方向180米', '类型': '车行门;区域门', '距离最近': True}, {'名称': '南门', '地址': '龙泽园街道西大街71号', '类型': '区域门'}]}, '地址': '风雅园二 区1号西南方向140米', '类型': '居民小区'}\n{'名称': '风雅园三区', '距离': '5公里外', '区划': '昌平区', '子节点': {'出入口': [{'名称': '西门', '地址': '风雅园三区3号楼正南方向20米', '类型': '车行门;区域门'}, {'名称': '西北门', '地址': '龙禧三街与育知路交叉口西160米', '类型': '区域门'}, {'名称': '南门', '地址': '风雅园三区', '类型': '区域门;车行门', '距离最近': True}, {'名称': '南1门', '地 址': '回龙观西大街85西北方向60米', '类型': '区域门;人行门'}, {'名称': '南2门', '地址': '回龙观西大街育知路1-1号中国邮政储蓄银行隔壁西北方向100米', '类型': '区域门;人行门'}, {'名称': '北门', '地址': '风雅园二区南门正南方向50米', '类型': '人行门;区域门'}], '停车场': [{'名称': '停车场', '地址': '风雅园三区3号楼东南方向20米', '类型': '地上露天停车场'}, {'名称': '停车场', '地址': '育知路风雅 园三区13号', '类型': '地下停车场', '距离最近': True}]}, '地址': '风雅园三区3号楼正东方向170米', '类型': '居民小区'}\n{'名称': '风雅园二区', '距离': '5公里外', '区划': '昌平区', '子节点': {'出入口': [{'名称': '东门', '地址': '风雅园二区1号正西方向40米', '类型': '车行门;区域门'}, {'名称': '东南门', '地址': '龙禧三街辅路与育知西路辅路交叉口正东方向102米', '类型': '区域门', '距离最近': True}, {'名称': '西门', '地址': '风雅园二区1号正西方向180米', '类型': '车行门;区域门'}, {'名称': '南门', '地址': '风雅园二区1号正南方向140米', '类型': '车行门;区域门'}, {'名称': '北门', '地址': '风雅 园一区18号楼底商东南方向60米', '类型': '区域门;车行门'}], '停车场': [{'名称': '停车场', '地址': '风雅园一区18号楼底商东南方向60米', '类型': '地上露天停车场'}, {'名称': '地下停车场', '地址': '风雅园二区1号西北方向60米', '类型': '地下停车场', '距离最近': True}]}, '地址': '风雅园二区1号正西方向130米', '类型': '居民小区'}\n{'名称': '风雅园一区', '距离': '5公里外', '区划': '昌平区', '子节点': {'出入口': [{'名称': '西南门', '地址': '风雅园一区18号楼底商正西方向50米', '类型': '区域门;车行门'}, {'名称': '南门', '地址': '回南北路68号正南方向180米', '类型': '车行门;区域门', '距离最近': True}], '停车场': [{'名称': '地下停车场', '地址': '回南北路68号西南方向150米', '类型': '地下停车场', '距离最近': True}]}, '地址': '回南北路68号东南方向140米', '类型': '居民小区'}\n{'名称': '风格雅 园', '距离': '5公里外', '区划': '朝阳区', '子节点': {'出入口': [{'名称': '西南门', '地址': '南湖南路8', '类型': '车行门;区域门'}, {'名称': '西北门', '地址': '南湖南路8号', '类型': '区域门;人行门', '距离最近': True}, {'名称': '北门', '地址': '南湖南路8号风格雅园', '类型': '区域门'}], '停车场': [{'名称': '停车场', '地址': '南湖南路8号', '类型': '地上露天停车场', '距离最近': True}, {'名称': '停车场', '地址': '南湖南路8号', '类型': '地下停车场'}]}, '地址': '南湖南路8号', '类型': '居民小区'}]\n\nbot: "
    ]
predictor.set_scene(Scenes.district)
print(predictor.predict([input_preprocess(item) for item in case]))
predictor.set_scene(Scenes.community)
print(predictor.predict([input_preprocess(text) for text in all_texts]))

case = "bot: 请问您要去哪里?\n\nuser:北师大\n\nbot: 前文user回复中有地址实体:北师大, 需要调知识库获取相关信息。\n\n知识库: [{'名称': '北京师范大学(南院)', '距离': '5公里内', '区划': '海淀区', '标签': '分校区;985大学;211大学;世界一流学科;世界一流大学', '地址': '学院南路12号', '类型': '知名大学'}\n{'名称': '北京师范大学(海淀校区)', '距离': '5公里内', '区划': ' >海淀区', '标签': '主校区;世界一流学科;985大学;211大学;世界一流大学', '子节点': {'出入口': [{'名称': '东门', '地址': '新街口外大街19', '类型': '区域门;车行门'}, {'名称': '东南门', ' 地址': '新街口外大街19-20', '类型': '区域门;车行门'}, {'名称': '西门', '地址': '新街口外大街19西门', '类型': '区域门;车行门'}, {'名称': '西门', '地址': '新街口外大街19号', '类型': '区域门;人行门'}, {'名称': '西1门', '地址': '新街口外大街19', '类型': '区域门;人行门', '距离最近': True}, {'名称': '西南门', '地址': '新街口外大街19旁门', '类型': '区域门;车行门'}, {'名称': '西北门', '地址': '新街口外大街19西北门', '类型': '车行门;区域门'}, {'名称': '南门', '地址': '新街口外大街19南门', '类型': '区域门;车行门'}, {'名称': '北门', '地址': '新街口外大街19北门', '类型': '区域门;车行门'}], '停车场': [{'名称': '停车场', '地址': '新街口外大街19', '类型': '地上露天停车场', '距离最近': True}, {'名称': '地下停车场', '地址': '新街口外大街19', '类型': '地下停车场'}]}, '地址': '新街口外大街19', '类型': '知名大学'}\n{'名称': '北京师范大学(新校区西区)', '距离': '5公里外', '区划': '昌平区', '标签': '分校区;世界一流学科;211大学;985大学;世界一流大学', '子节点': {'出入口': [{'名称': '东门', '地址': '沙河高教园南三街9号正南方向180米', '类型': '车行门'}, {'名称': '东北门', '地址': '沙河地区北沙河 西三路西南侧', '类型': '区域门'}, {'名称': '北门', '地址': '高教园区南二街9号正东方向50米', '类型': '车行门;区域门'}, {'名称': '北门', '地址': '满井路与北沙河西三路交叉口西南角', ' 类型': '区域门', '距离最近': True}]}, '地址': '南三街路口正西方向190米', '类型': '知名大学'}\n{'名称': '北京师范大学(西城校区)', '距离': '5公里外', '区划': '西城区', '标签': '分校区;世界一流学科;211大学;985大学;世界一流大学', '子节点': {'出入口': [{'名称': '南门', '地址': '', '类型': '区域门', '距离最近': True}]}, '地址': '定阜街1号', '类型': '知名大学'}]\n\nbot:"
predictor.set_scene(Scenes.school)
print(predictor.predict([case], scene=Scenes.school))

# NER 模块
# ner_model = predictors.get(Scenes.district)  # NER 暂时先选一个场景模型，因为语料中是有ner相关的语料的。 后期需要提高效率时，再做一个轻量级的模型。
# 类别判别器，前期使用正则，后期需要训练一个小模型
classifier = get_classifier


pattern_district_confirm = re.compile("就[是要].*?去")


def predict(query, chatbot, history):
    if predictor.scene == Scenes.district and not pattern_district_confirm.search(query):
        history = [f"bot: 请问您要去哪里?\n\n"]

    if not chatbot or len(history) == 1:
        prompt = ""
        for i, sentence in enumerate(history):
            prompt += sentence
        input_texts = [prompt + f"user: {query}"]
        input_texts = [input_preprocess(text) for text in input_texts]
        history.append("user:" + query + "\n\n")

        print("intent-input:", input_texts)
        predictor.set_scene(Scenes.district)  # NER 暂时先选一个场景模型，因为语料中是有ner相关的语料的。
        output = predictor.predict(input_texts)
        response = output["result"][0]  # 跟插件结合起来
        history.append("bot: " + response + "\n\n")
        print("intent-output:", output)

        res = plugin_pattern.search(response)
        kg = None
        if res:
            entity = res.group("name")
            kg = search(entity)
            category = classifier(kg)
            predictor.set_scene(category)
        chatbot, history = inner_predict(query, chatbot, history, predictor, kg=kg)
    else:
        chatbot, history = inner_predict(query, chatbot, history, predictor)
    return chatbot, history


def inner_predict(query, chatbot, history, predictor, kg=None):
    kg_str = "\n".join([f"{item}" for item in kg]) if kg is not None else None
    prompt = ""
    for i, sentence in enumerate(history):
        prompt += sentence

    if kg_str is None:
        input_texts = [prompt + f"user: {query}" + "\n\nbot: "]
        history.append(f"user: {query}" + "\n\n")
    else:
        input_texts = [prompt + f"知识库: [{kg_str}]" + "\n\nbot: "]
        history.append(f"知识库: [{kg_str}]" + "\n\n")
    input_texts = [input_preprocess(text) for text in input_texts]

    print("input:", input_texts)
    output = predictor.predict(input_texts)
    response = output["result"][0]
    print("output:", output)

    history.append("bot: " + response + "\n\n")
    chatbot.append((query, ""))
    chatbot[-1] = (query, parse_text(response))

    print(f"chatbot is {chatbot}")
    print(f"history is {history}")

    return chatbot, history


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [f"bot: 请问您要去哪里?\n\n"]


gr.Chatbot.postprocess = postprocess


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center"> 导航小助手 </h1>""")
    gr.Markdown(" ###    请问您要去哪里？")
    chatbot = gr.Chatbot()
    history = gr.State([f"bot: 请问您要去哪里?\n\n"])  # (message, bot_message)

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=4).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")

    submitBtn.click(predict, [user_input, chatbot, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(server_name="0.0.0.0", server_port=35001, share=False, inbrowser=True)
