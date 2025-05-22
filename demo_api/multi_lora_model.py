import os
import json
from copy import deepcopy
import paddle
from paddle.distributed import fleet

from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.peft.prefix import (
    chatglm_pad_attention_mask,
    chatglm_postprocess_past_key_value,
)
from paddlenlp.transformers import (
    ChatGLMConfig,
    ChatGLMForConditionalGeneration,
    ChatGLMTokenizer,
)


def load_base_model(dtype, model_fp):
    tensor_parallel_degree = paddle.distributed.get_world_size()
    tensor_parallel_rank = 0
    if tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        tensor_parallel_rank = hcg.get_model_parallel_rank()

    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_fp,
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_rank=tensor_parallel_rank,
        # load_state_as_np=True,
        dtype=dtype,
    )
    model.eval()
    return model


def load_json(fp):
    with open(fp, "r") as f:
        data = json.load(f)
        return data


def load_predictors(args):
    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_name_or_path)

    scenes = [name for name in os.listdir(args.lora_dir)]
    valid_scenes = args.scenes_name
    for scene in valid_scenes:
        if scene not in scenes:
            raise ValueError(f"scene={scene} not in args.lora_dir!")

    lora_config = LoRAConfig.from_pretrained(os.path.join(args.lora_dir, Scenes.school, "online_model"))  # 所有的lora-model 都是相同的dtype
    base_model = load_base_model(lora_config.dtype, args.model_name_or_path)

    scenes_lora_path = {
        Scenes.school: os.path.join(args.lora_dir, Scenes.school, "online_model"),
        Scenes.hospital: os.path.join(args.lora_dir, Scenes.hospital, "online_model"),
        Scenes.community: os.path.join(args.lora_dir, Scenes.community, "online_model"),
        Scenes.district: os.path.join(args.lora_dir, Scenes.district, "online_model"),
        Scenes.transportation: os.path.join(args.lora_dir, Scenes.transportation, "online_model"),
    }

    # lora_model_school = LoRAModel(base_model, load_json(os.path.join(args.lora_dir, Scenes.school, "online_model", "lora_config.json")))
    # restored_model = lora_model_school.restore_original_model()
    # restored_model.eval()
    # print(f"restored_model={type(restored_model)}, base_model={type(base_model)}")
    # lora_model_hospital = LoRAModel(restored_model,  load_json(os.path.join(args.lora_dir, Scenes.hospital, "online_model", "lora_config.json")))
    # print(lora_model_school.parameters)
    # assert lora_model_school.parameters() != lora_model_hospital.parameters()
    # restored_model = lora_model_hospital.restore_original_model()
    # restored_model.eval()
    # lora_model_community = LoRAModel(restored_model, load_json(os.path.join(args.lora_dir, Scenes.community, "online_model", "lora_config.json")))
    # restored_model = lora_model_community.restore_original_model()
    # restored_model.eval()
    # lora_model_district = LoRAModel(restored_model, load_json(os.path.join(args.lora_dir, Scenes.district, "online_model", "lora_config.json")))
    # restored_model = lora_model_district.restore_original_model()
    # restored_model.eval()
    # lora_model_transportation = LoRAModel(restored_model, load_json(os.path.join(args.lora_dir, Scenes.transportation, "online_model", "lora_config.json")))
    # school_predictor = Predictor(args, tokenizer, lora_model_school)
    # hospital_predictor = Predictor(args, tokenizer, lora_model_hospital)
    # community_preditor = Predictor(args, tokenizer, lora_model_community)
    # district_preditor = Predictor(args, tokenizer, lora_model_district)
    # transportation_preditor = Predictor(args, tokenizer, lora_model_transportation)

    predictor = Predictor(args, tokenizer, base_model, scenes_lora_path)
    return predictor


class Scenes:
    def __int__(self):
        pass
    hospital = "hospital"
    school = "school"
    community = "community"
    transportation = "transportation"
    district = "district"


class Predictor:
    def __init__(self, args, tokenizer, base_model, scenes_lora_path):
        self.tokenizer = tokenizer
        self.batch_size = args.batch_size
        self.src_length = args.src_length
        self.tgt_length = args.tgt_length
        self.base_model = base_model
        self.scene = None
        self.model = None
        self.scenes_lora_path = scenes_lora_path

        # self.model = LoRAModel.from_pretrained(base_model, lora_path)  # 这个时候会更新self.base_model
        # self.model.mark_only_lora_as_trainable()
        # # self.model = model
        # self.model.eval()

    def set_scene(self, scene):
        if scene is None and self.scene is None:
            print("scene and self.scene are not None!")
        if scene != self.scene:
            self.scene = scene
            if scene is None:
                self.model = None
            else:
                if self.model is not None:
                    self.base_model = self.model.restore_original_model()
                self.model = LoRAModel.from_pretrained(self.base_model,
                                                       self.scenes_lora_path.get(scene))  # 这个时候会更新self.base_model
                self.model.mark_only_lora_as_trainable()
                self.model.eval()

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            padding=True,
            max_length=self.src_length,
            truncation=True,
            truncation_side="left",
        )
        inputs_tensor = {}
        for key in inputs:
            inputs_tensor[key] = paddle.to_tensor(inputs[key])
        return inputs_tensor

    def infer(self, inputs):
        result = self.model.generate(
            **inputs,
            decode_strategy="sampling",
            top_k=1,
            max_length=self.tgt_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        result = result[0]
        return result

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            res = res.strip("\n")
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts, scene=None):
        # if scene is None and self.scene is None:
        #     raise ValueError(f"one of scene and self.scene is not None!")
        # if scene is not None and scene != self.scene:
        #     self.scene = scene
        #     if self.model is not None:
        #         self.base_model = self.model.restore_original_model()
        #     self.model = LoRAModel.from_pretrained(self.base_model, self.scenes_lora_path.get(scene))  # 这个时候会更新self.base_model
        #     self.model.mark_only_lora_as_trainable()
        #     self.model.eval()

        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output
