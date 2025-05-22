import grequests
import json
import re
import math
import random
import copy
from decimal import Decimal
from collections import defaultdict


class SubNode:
    def __init__(self):
        pass
    gate = "出入口"
    park = "停车场"
    terminal = "航站楼"
    entrance = "进站口"


road_pattern = [re.compile(r"(?P<road>.*?(?:街口|路口|交汇处|交叉口|大厦|公园|家园|胡同|广场))"),
                re.compile(r"(?P<road>.*?[路街道村])"),
                re.compile(r"(?P<road>.*?(?:花园))")]
# branch_pattern = re.compile(r"\((?P<branch>.*?)\)$")
# community_branch_pattern = re.compile(r"^(?P<main>.*?)(?P<branch>[一二三四五六七八九\d][期区])")
# school_pattern = re.compile(r"(?:学校|小学|中学|大学|学院)$")
# community_pattern = re.compile(r"(?:小区|社区)$")
# campus_pattern = re.compile("^(?P<main>.*?(?:学校|小学|中学|大学|学院|分校))(?P<branch>.*?(?:校区|院区|校园))$")
# branch_campus_pattern = re.compile("(?P<campus>.*?(?:校区|院区))")  # 括号内的部分再做匹配


def add_closest_feature(items, target_name=None, target_idx=None):
    new_items = []
    if target_name is not None:
        for item in items:
            if item["名称"] == target_name:
                item["距离最近"] = True
            new_items.append(item)
    elif target_idx is not None:
        for idx, item in enumerate(items):
            if idx == target_idx:
                item["距离最近"] = True
            new_items.append(item)

    return new_items


def remove_distance_feature(items):
    new_items = copy.deepcopy(items)
    for item in new_items:
        if "距离" in item:
            item.pop("距离")

    return new_items


def convert_distance_range(items):
    new_items = copy.deepcopy(items)
    for item in new_items:
        if "距离" in item:
            distance = item["距离"] / 1000
            if distance < 2:
                item["距离"] = "2公里内"
            elif distance < 5:
                item["距离"] = "5公里内"
            else:
                item["距离"] = "5公里外"

    return new_items


def process_address(addr):
    if len(addr) > 20 and "," in addr:
        address = addr.split(",")
        tmp = ""
        for elem in address:
            if len(tmp + "," + elem) > 20:
                break
            tmp += "," + elem
        return tmp.strip(",") + "等"
    if "(" in addr:
        addr = addr.split("(")[0]
    return addr


def request_poi_infos(urls, topn=10):
    req_list = (grequests.get(url) for url in urls)
    pois_infos = []

    for resp in grequests.map(req_list, size=10):
        poi_result = []
        poi_name_district = set()  # 去重复数据
        try:
            res = json.loads(resp.text)
        except AttributeError:
            pois_infos.append(poi_result)
            continue
        # print('resp.text:', res)
        if res["msg"] != "成功" or "data" not in res:
            pois_infos.append(poi_result)
            continue
        if "pois" not in res["data"]:
            if "dist" not in res["data"]:
                pois_infos.append(poi_result)
            else:
                dist = res["data"]["dist"][0]
                # parents = dist["parents"]
                name = dist["name"]
                poi_result.append({"名称": name, "类型": "行政区划"})
                pois_infos.append(poi_result)
                continue

        pois = res["data"]["pois"]
        if not pois:
            pois_infos.append(poi_result)
            continue

        chosen_num = topn  # 选择topn个结果，query太泛时 需要根据road进行再次选定
        for i in range(min(chosen_num, len(pois))):
            chosen = pois[i]
            # print(chosen["distance"])
            distance = chosen["distance"]  # 有distance="" 的情况
            if not distance:
                distance = random.uniform(1000, 2000)
            distance = float(Decimal(distance).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP"))
            tmp = {"名称": chosen["name"], "距离": distance}
            district = chosen["districtName"]
            if district:
                tmp['区划'] = district
            tag_name = chosen["tagName"]
            if tag_name:
                tmp["标签"] = tag_name
            sub_poi = chose_sub_poi(chosen["children"], distance) if "children" in chosen else None
            if sub_poi:
                tmp["子节点"] = sub_poi
            addr = process_address(chosen["address"])
            if addr:
                tmp["地址"] = addr
            sub_type = chosen['subType']
            if sub_type:
                tmp["类型"] = sub_type

            if chosen["name"] + chosen["districtName"] not in poi_name_district:
                poi_result.append(tmp)
                poi_name_district.add(chosen["name"] + chosen["districtName"])

        pois_infos.append(poi_result)

    assert len(urls) == len(pois_infos)
    return pois_infos


def chose_sub_poi(children, distance):
    """解析 某些类型的子节点

       子节点的distance字段是无值的。在poi的距离值上 随机加减一个小的值。
    """
    valid_subcate = set([SubNode.gate, SubNode.terminal, SubNode.park, SubNode.entrance])
    # print(f"children={children}")
    subcategory_candi = defaultdict(list)
    name_tag_name = set()  # 去重复
    for dic in children:
        # print('tnm', dic['tnm'])
        # if dic["tnm"] == "座":
        #     print(dic)
        subcategory = dic["tnm"]
        if subcategory not in valid_subcate:
            continue
        # print('tnm', dic['tnm'])
        pois = dic["pois"]

        for poi in pois:
            name = poi["shortName"] if poi["shortName"] else poi["name"]
            # if subcategory == SubNode.park:
            #     name = poi["name"]
            # tmp = {"名称": name, "道路": process_address(get_road_from_addr(poi["address"]))}  # 道路 改回地址
            tmp = {"名称": name, "地址": process_address(poi["address"])}
            if subcategory == SubNode.gate:
                if "人行门" not in poi["tagName"] and "区域门" not in poi["tagName"] and "车行门" not in poi["tagName"]:  # 这个条件更严格了
                    continue
                if poi["mainDoor"]:  # 只保留正门是true的情况
                    tmp["正门"] = poi["mainDoor"]
                tmp["类型"] = poi["tagName"]  # if poi["tagName"] in {"车行门", "人行门", "区域门"}
            elif subcategory == SubNode.park:
                tmp["类型"] = poi["categoryName"]  # 地下停车场、地上停车场

            if "类型" in tmp:
                if name + tmp["类型"] in name_tag_name:
                    continue
                name_tag_name.add(name + tmp["类型"])

            poi_distance = poi["distance"]
            if poi_distance:
                tmp["距离"] = float(poi_distance)
            else:
                tmp["距离"] = distance + random.uniform(-3, 3)
                # # FIXME
                # tmp["距离"] = random.uniform(1000, 2000)
            tmp["距离"] = float(Decimal(tmp["距离"]).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP"))

            if tmp not in subcategory_candi[subcategory]:
                subcategory_candi[subcategory].append(tmp)
    # print(f"subcategory_candi={subcategory_candi}")
    result = {}
    for key, value in subcategory_candi.items():
        val_distances = [i['距离'] for i in value]
        min_idx = val_distances.index(min(val_distances))
        value_wo_distance = []
        for idx, elem in enumerate(value):
            if idx == min_idx:
                elem["距离最近"] = True
            elem.pop("距离")
            value_wo_distance.append(elem)
        result[key] = value_wo_distance

    return dict(result) if result else None


def convert(x, y):
    x = x / 20037508.34 * 180
    y = y / 20037508.34 * 180

    y = 180 / math.pi * (2 * math.atan(math.exp(y * math.pi / 180)) - math.pi / 2)
    return x, y


pattern_school = re.compile(r"[主分]校区")
pattern_hospital = re.compile(r"[总分医]院")


def search(entity, city="北京市"):
    # beijing坐标（ 上海市：13521987.7264823,3641085.27444297
    # x, y = 12955994.0911619, 4824868.85787938
    # x, y = convert(x, y)

    x, y = 116.34539831208934, 39.99613351536769

    url = f"http://10.130.40.137:10003/service-mobile-23m/navisearch/keywords?keywords={entity}&location={x},{y}&city={city}&pageIndex=1&pageSize=10&useCache=0&debug=true&children=1"

    items = request_poi_infos([url], topn=10)[0]
    # print('items:', items)
    if not items:
        return items
    # # 主校区、分校区；总院、分院
    # items_new = []
    # flag = True if "校区" in items[0]["类型"] or "院区" in items[0]["类型"] else False  # 训练语料中有错误
    # for item in items:
    #     tag_name = item["类型"]
    #     if flag:
    #         if "校区" not in tag_name and "院区" not in tag_name:
    #             continue
    #     if "主校区" in tag_name and "校区" not in item["名称"]:
    #         item["名称"] += f"(主校区)"
    #     if "总院" in tag_name and "院)" not in item["名称"]:
    #         item["名称"] += f"(总院)"
    #     items_new.append(item)
    # return items_new[:5]

    # 片区的处理
    district_subtype = {"商圈", "道路", "立交桥"}
    items_new_district = [item for item in items if item["类型"] in district_subtype and entity == item["名称"]]
    items = items_new_district if items_new_district else items

    candidate = items[0]

    tag_name = candidate["标签"] if "标签" in candidate else None
    sub_type = candidate["类型"] if "类型" in candidate else None

    min_idx = 0  # 虚假数据
    if sub_type:
        if sub_type == "居民小区":
            min_idx = 1
            items = [item for item in items if item.get("类型", None) == sub_type][:5]
        else:
            items = [item for item in items if item.get("类型", None) == sub_type or check_tag_name(tag_name, item.get("标签", None))][:5]
    else:
        items = [item for item in items if check_tag_name(tag_name, item.get("标签", None))][:5]

    # items = add_closest_feature(items, target_idx=min_idx)
    items = convert_distance_range(items)

    return items


def check_tag_name(tag_name_1, tag_name_2):
    if tag_name_1 is None or tag_name_2 is None :
        return False

    def is_hospital_or_school(tag_name_1, tag_name_2):
        for pattern in [pattern_hospital, pattern_school]:
            if pattern.search(tag_name_1) and pattern.search(tag_name_2):
                return True
        return False
    is_same = is_hospital_or_school(tag_name_1, tag_name_2)
    if is_same:
        return True
    return False

