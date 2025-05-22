from multi_lora_model import Scenes


def get_classifier(kg):
    if not kg:
        return Scenes.community

    category_keywords = {
        Scenes.school: {"学校", "大学", "校区", "学院"},
        Scenes.hospital: {"医院", "总院", "分院"},
        Scenes.community: {"小区"},
        Scenes.district: {"商圈", "道路", "立交桥", "行政区划"},
        Scenes.transportation: {"机场", "火车站"},
    }
    if not kg:
        return None

    def get(kind, label, name):
        for category, kws in category_keywords.items():
            flag = False
            for kw in kws:
                if kw in kind or kw in label or name.endswith(kw):
                    flag = True
                    break

            if flag:
                return category
        return None

    candidate = kg[0]
    kind = candidate["类型"]
    label = candidate["标签"] if "标签" in candidate else ""
    name = candidate["名称"]

    category = get(kind, label, name)
    if category is None:  # 普通场景的结果，默认先使用小区的模型
        category = Scenes.community

    return category

