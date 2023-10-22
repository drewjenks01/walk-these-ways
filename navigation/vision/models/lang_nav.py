# Use a pipeline as a high-level helper
import torch.nn as nn

from transformers import AutoConfig, LlamaConfig 
from llava.llava.model.builder import load_pretrained_model

model_path = 'liuhaotian/llava-v1.5-13b'
model_base = None
model_name = ''
load_8bit = False
load_4bit = True
device='mps'

if model_path.endswith("/"):
    model_path = model_path[:-1]
if model_name is None:
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        model_name = model_paths[-2] + "_" + model_paths[-1]
    else:
        model_name = model_paths[-1]
else:
    model_name = model_name
tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name, load_8bit, load_4bit, device=device)



def get_model_size(model: nn.Module, data_width=16, group_size=-1):

    if group_size != -1:
        data_width += (16 + 4) / group_size

    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

# Evaluate the model
model_size = get_model_size(model, data_width=32, group_size=128)
print(f"model size: {model_size/MiB:.2f} MiB")