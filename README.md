# navigation-assistant

	基于gradio 部署导航助手demo


### 多模型实现多场景的多轮对话

	启动服务: sh run.sh

### 模型的实现细节及模型效果




### 默认的测试地址
    
	http://10.130.200.250:35001/ 在服务启动后可进行输入文本测试效果。

### 测试demo的注意事项

	1. 为简化工作量而使用的策略：
		
		1. NER(目的地抽取)模型：使用的是正则做的，所以需要输入较固定的句式（如：我要去北京大学、去北师大等）；
		2. 场景判别器：对`搜索引擎返回的知识库`进行关键词匹配后获取场景类别；

	2. 由于搜索引擎url需要传入经纬度，而这个参数目前不方便获取，故代码中固定了北京的经纬度。所以，测试时请测试北京的poi（目的地）。
		1. 见代码中https://gitlab.navinfo.com/ni_engine_search/search-source/model-train-search/navigation_assistant_demo/-/blob/master/demo_api/pluginer.py#L235

### 一条完整的测试样例
      
      - "bot": "请问您要去哪里？"
      - "user": "去北大三院。
      - "知识库知识": [{'名称': '北京大学第三医院', '距离': '5公里内', '区划': '海淀区', '标签': '三级甲等医院;三级医院;总院', '子节点': {'出入口': [{'名称': '东门', '地址': '花园北路49号', '类型': '车行门;区域门'}, {'名称': '东南门', '地址': '花园北路49号', '类型': '车行门;区域门', '距离最近': True}, {'名称': '西门', '地址': '花园北路49号', '类型': '车行门;区域门'}, {'名称': '南门', '地址': '花园北路49旁门', '类型': '车行门;区域门'}, {'名称': '南1门', '地址': '花园北路49号', '类型': '区域门;人行门'}, {'名称': '南2门', '地址': '花园北路49号', '类型': '车行门;区域门'}, {'名称': '北门', '地址': '花园北路49号', '类型': '人行门;区域门'}, {'名称': '北门', '地址': '花园北路49号', '类型': '车行门;区域门'}], '停车场': [{'名称': '立体停车库', '地址': '花园北路49号', '类型': '地上非露天停车场'}, {'名称': '停车场', '地址': '花园北路49旁门', '类型': '地上露天停车场'}, {'名称': '3区停车场', '地址': '花园北路49号', '类型': '地上露天停车场'}, {'名称': '1区停车场', '地址': '花园北路49号', '类型': '地下停车场'}, {'名称': '急诊停车场', '地址': '花园北路49号', '类型': '地下停车场', '距离最近': True}]}, '地址': '花园北路49号', '类型': '三级医院', '距离最近': True}, {'名称': '北京大学第三医院海淀院区', '距离': '5公里外', '区划': '海淀区', '标签': '三级医院', '子节点': {'出入口': [{'名称': '东门', '地址': '中关村大街29', '类型': '车行门', '距离最近': True}, {'名称': '西南门', '地址': '中关村大街29号', '类型': '人行门'}, {'名称': '南门', '地址': '中关村大街29', '类型': '车行门'}], '停车场': [{'名称': '停车场', '地址': '中关村大街29', '类型': '地上露天停车场', '距离最近': True}]}, '地址': '中关村大街29', '类型': '三级医院'}, {'名称': '北京大学第三医院(首都国际机场院区)', '距离': '5公里内', '区划': '朝阳区', '标签': '二级医院;分院;二级甲等医院', '子节点': {'出入口': [{'名称': '东门', '地址': '岗山路9', '类型': '车行门;区域门'}, {'名称': '东南门', '地址': '机场南路东里17号楼大学第三医院首都国际机场院区内', '类型': '区域门;车行门'}, {'名称': '东北门', '地址': '岗山路', '类型': '区域门'}, {'名称': '西门', '地址': '机场南路与西平街交叉口正东方向92米', '类型': '区域门', '距离最近': True}, {'名称': '南门', '地址': '岗山路9', '类型': '区域门;车行门'}], '停车场': [{'名称': '停车场', '地址': '岗山路9', '类型': '地上露天停车场', '距离最近': True}]}, '地址': '岗山路9', '类型': '一般医院'}, {'名称': '兵器工业北京北方医院', '距离': '5公里内', '区划': '海淀区', '标签': '一级医院', '子节点': {'停车场': [{'名称': '停车场', '地址': '车道沟10号', '类型': '地上露天停车场'}, {'名称': '地下停车场', '地址': '车道沟10号院', '类型': '地下停车场', '距离最近': True}]}, '地址': '车道沟10号院', '类型': '一般医院'}, {'名称': '北京大学第三医院(延庆医院)', '距离': '5公里外', '区划': '延庆区', '标签': '分院;二级甲等医院', '子节点': {'出入口': [{'名称': '东门', '地址': '东顺城街28号', '类型': '出入口;区域门'}, {'名称': '东北门', '地址': '东顺城街28号', '类型': '车行门;区域门'}, {'名称': '西门', '地址': '东顺城街28号', '类型': '出入口;区域门', '距离最近': True}, {'名称': '南门', '地址': '东顺城街28号', '类型': '区域门;车行门'}, {'名称': '北门', '地址': '东顺城街28号', '类型': '车行门;区域门'}]}, '地址': '东顺城街28号', '类型': '一般医院'}]
      - "bot": "由知识库信息知：北京大学第三医院有总院、海淀院区、首都国际机场院区、延庆医院。由知识库中poi标签信息和距离信息知：2公里内无总院或分院，5公里内有总院。您要去北京大学第三医院总院吗？"
      - "user": "是的"
      - "bot": "好的，北京大学第三医院总院位于花园北路，将为您导航目的地为北京大学第三医院总院。由知识库中子节点信息知，北京大学第三医院有出入口、停车场信息，将进一步确认目的地。您去门口还是停车场？"
      - "user": "门口"
      - "bot": "根据知识库中出入口信息，北京大学第三医院有东门、东南门、西门、南门、南1门、南2门、北门这7个出入口，最近的是东南门。去东南门可以吗？"
      - "user": "选个可以行人的吧。"
      - "bot": "好的，根据知识库中出入口的类型信息，南1门是类型为人行门的出入口，将为您导航目的地为北京大学第三医院总院南1门。" 

