from dataclasses import dataclass, field
import os
from os.path import isdir, isfile
from pathlib import Path
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import AutoPeftModelForCausalLM
from auto_gptq import exllama_set_max_input_length

@dataclass
class GptqConfig:
    ckpt: str = field(
        default=None,
        metadata={
            "help": "Load quantized model. The path to the local GPTQ checkpoint."
        },
    )
    wbits: int = field(default=16, metadata={"help": "#bits to use for quantization"})
    groupsize: int = field(
        default=-1,
        metadata={"help": "Groupsize to use for quantization; default uses full row."},
    )
    act_order: bool = field(
        default=True,
        metadata={"help": "Whether to apply the activation order GPTQ heuristic"},
    )


def load_gptq_quantized(model_name, gptq_config: GptqConfig):
    print("Loading GPTQ quantized model...")

    try:
        script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        module_path = os.path.join(script_path, "../repositories/GPTQ-for-LLaMa")

        sys.path.insert(0, module_path)
        from llama import load_quant
    except ImportError as e:
        print(f"Error: Failed to load GPTQ-for-LLaMa. {e}")
        print("See https://github.com/lm-sys/FastChat/blob/main/docs/gptq.md")
        sys.exit(-1)

    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
    )
    tokenizer.pad_token = "</s>"
    tokenizer.add_prefix_space = False

    # only `fastest-inference-4bit` branch cares about `act_order`
    if gptq_config.act_order:
        model = load_quant(
            model_name,
            find_gptq_ckpt(gptq_config),
            gptq_config.wbits,
            gptq_config.groupsize,
            act_order=gptq_config.act_order,
        )
    else:
        # if "checkpoint" in model_name:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            # low_cpu_mem_usage=True,
            # return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # else:
        #     # other branches
        #     # model = load_quant(
        #     #     model_name,
        #     #     find_gptq_ckpt(gptq_config),
        #     #     gptq_config.wbits,
        #     #     gptq_config.groupsize,
        #     # )
        #     model = AutoModelForCausalLM.from_pretrained(
        #         model_name,
        #         device_map='auto',
        #         trust_remote_code=False,
        #         revision="main",
        #         # attn_implementation="flash_attention_2"
        #     )

        # # model = exllama_set_max_input_length(model, max_input_length=4096)
        model = exllama_set_max_input_length(model, max_input_length=16384)

    return model, tokenizer


def find_gptq_ckpt(gptq_config: GptqConfig):
    if Path(gptq_config.ckpt).is_file():
        return gptq_config.ckpt

    for ext in ["*.pt", "*.safetensors"]:
        matched_result = sorted(Path(gptq_config.ckpt).glob(ext))
        if len(matched_result) > 0:
            return str(matched_result[-1])

    print("Error: gptq checkpoint not found")
    sys.exit(1)
