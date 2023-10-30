# Use a pipeline as a high-level helper
import torch.nn as nn
import random
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn

from navigation.vision.models.minigpt4.minigpt4.common.config import Config

from navigation.vision.models.minigpt4.minigpt4.common.registry import registry
from navigation.vision.models.minigpt4.minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from navigation.vision.models.minigpt4.minigpt4.datasets.builders import *
from navigation.vision.models.minigpt4.minigpt4.models import *
from navigation.vision.models.minigpt4.minigpt4.processors import *
from navigation.vision.models.minigpt4.minigpt4.runners import *
from navigation.vision.models.minigpt4.minigpt4.tasks import *

from navigation import constants


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark = False
cudnn.deterministic = True

class LangNav:
    def __init__(self):
        self.chat, self.CONV_VISION = self._initialize_chat()
        self.chat_state = self.CONV_VISION.copy()

        self.bounding_box_size = 100
        self.temperature = 0.6  # 0.1-1.5

        self.context = []

    def _initialize_chat(self):
        args = parse_args()
        cfg = Config(args)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(constants.DEVICE)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        model = model.eval()

        CONV_VISION = Conversation(
            system="",
            roles=(r"<s>[INST] ", r" [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )

        chat = Chat(model, vis_processor, constants.DEVICE)

        return chat, CONV_VISION

    def ask(self, message, image):
        image_list = []
        self.chat.upload_img(image, self.chat_state,image_list)
        self.chat.encode_img(image_list)
        self.chat.ask(message, self.chat_state)
        answer = self.chat.answer(
            conv=self.chat_state,
            img_list=image_list,
            temperature=self.temperature,
            max_new_tokens=500,
            max_length=2000
        )[0]

        context_response = self.chat.ask(self._get_context_prompt(), self.chat_state)
        self._add_to_context(context_response)
        return answer
    
    def _get_model_size(model: nn.Module, data_width=16, group_size=-1):
        Byte = 8
        KB = 1000 * Byte
        MB = 1000 * KB
        GB = 1000 * MB

        if group_size != -1:
            data_width += (16 + 4) / group_size

        num_elements = 0
        for param in model.parameters():
            num_elements += param.numel()
        return f"model size: {num_elements * data_width/GB:.2f} GB"
    
    def _get_context_prompt(self):
        context_prompt = ''
        return context_prompt
    
    def _add_to_context(self, context_response):
        self.context.append(context_response)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args