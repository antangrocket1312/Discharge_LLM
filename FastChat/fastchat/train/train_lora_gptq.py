# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import logging
import pathlib
import typing
import os

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training \
    # , cast_mixed_precision_params
import transformers
from transformers import Trainer, BitsAndBytesConfig, deepspeed
import torch

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    make_supervised_data_module,
)

from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# import os
# import sys
# module_path = os.path.join("./repositories/GPTQ-for-LLaMa")
# sys.path.insert(0, module_path)
# from llama import load_quant
import pandas as pd
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from auto_gptq import exllama_set_max_input_length


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def format_instruction(example):
    brief_hospital_course_prompt = """In this task, you are provided with a discharge summary delimited by triple quotes.
Discharge summaries are documents that outline the care a patient received during their hospital stay, including diagnoses, treatments, and follow-up care instructions, prepared at the time of a patient's discharge. 
Discharge summaries are split into various sections and written under a variety of headings, relating to admission, diagnosis and relevant discharge information. But the provided Discharge summary will be missing the “Brief Hospital Course”. “Brief Hospital Course” is a section of the discharge summaries that outlines the key events of a patient's hospital stay, including the progression from admission to discharge. It is written for the subsequent care providers about the critical aspects of the patient.
You are tasked to the missing “Brief Hospital Course” section in the discharge summary, based on the information of other sections in the discharge summary.
Brief Hospital Course outlines the key events of a patient's hospital stay, including the progression from admission to discharge. It is written for the subsequent care providers about the critical aspects of the patient

Perform the following actions to solve this task:
- Identify and extract the medical information from different sections from the discharge summaries relating to the patient’s admission, treatment and discharge.
- On extracted information from different sections, generate the content of 'Brief Hospital Course' section based on the main information across other sections, using abstractive summarization.

If possible, on every diagnoses being identified, please have a separate paragraph describing further details on the treatment history of the patient.

Discharge summary: 
\"\"\"%s\"\"\"

Brief Hospital Course:
\"\"\"%s\"\"\"
"""

    output_texts = []
    for i in range(len(example['processed_text'])):
        text = brief_hospital_course_prompt % (example['processed_text'][i], example['brief_hospital_course'][i])
        output_texts.append(text)
    return output_texts


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    # training_args.fp16 = True
    # training_args.bf16 = False
    print('local_rank', training_args.local_rank)

    if training_args.flash_attn:
        replace_llama_attn_with_flash_attn()

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # model = load_quant("./models/vicuna-7B-v1.5-GPTQ", "./models/vicuna-7B-v1.5-GPTQ/model.safetensors", 4, 128);
    if training_args.flash_attn:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # device_map=device_map,
            device_map='auto',
            trust_remote_code=False,
            # revision="main",
            revision="gptq-4bit-128g-actorder_True",
            attn_implementation="flash_attention_2"
        )
    else:
        # from accelerate import Accelerator
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        #     model_args.model_name_or_path, device_map={"": Accelerator().process_index}, revision="gptq-4bit-128g-actorder_True")

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # device_map=device_map,
            device_map='auto',
            trust_remote_code=False,
            # revision="main",
            revision="gptq-4bit-128g-actorder_True",
            # torch_dtype=torch.bfloat16
            # torch_dtype=torch.float16
            # quantization_config=BitsAndBytesConfig(
            #     load_in_4bit=False,
            #     bnb_4bit_use_double_quant=False,
            #     # bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=compute_dtype,
            # )
        )
    if training_args.model_max_length * training_args.per_device_train_batch_size >= 4096:
        model = exllama_set_max_input_length(model,
                                             max_input_length=training_args.model_max_length * training_args.per_device_train_batch_size)
    # model.to(dtype=torch.bfloat16, device=torch.device('cuda:0'))

    # model = AutoGPTQForCausalLM.from_quantized(
    #     model_args.model_name_or_path,
    #     model_basename='model',
    #     use_safetensors=True,
    #     trust_remote_code=True,
    #     device="cuda:0",
    #     use_triton=False,
    #     quantize_config=None)

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     device_map=device_map,
    #     quantization_config=BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=compute_dtype,
    #     )
    #     if lora_args.q_lora
    #     else None,
    # )
    #

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        # target_modules=["k_proj", "o_proj", "q_proj", "v_proj", "down_proj", "gate_proj", "up_proj"],
        # target_modules=["k_proj", "o_proj", "q_proj", "v_proj"],
        # target_modules=["v_proj", "o_proj"], # It generates ,,,,,,-_________\__\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\
        # target_modules=lora_args.lora_target_modules,
        # target_modules=["k_proj", "q_proj"],
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    model = get_peft_model(model, lora_config)
    # cast_mixed_precision_params(model, torch.float16)

    # if training_args.flash_attn:
    #     for name, module in model.named_modules():
    #         if "norm" in name:
    #             module = module.to(compute_dtype)
    #         if "lm_head" in name or "embed_tokens" in name:
    #             if hasattr(module, "weight"):
    #                 module = module.to(compute_dtype)
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    data_path = data_args.data_path
    if data_path.endswith("pkl"):
        df = pd.read_pickle(data_path)
        train = Dataset.from_pandas(df)
        # train = load_dataset("mlabonne/guanaco-llama2-1k", split="train").select(range(50))
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer, args=training_args,
            train_dataset=train,
            # dataset_text_field='text',
            dataset_text_field='train_text',
            # peft_config=lora_config,
            # formatting_func=format_instruction,
            # packing=True,
            max_seq_length=training_args.model_max_length
        )
    elif data_path.endswith("json"):
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        # model.params = model.to_bf16(model.params)

        trainer = Trainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )

    model.config.use_cache = False

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # trainer.save_model()
    trainer.save_state()

    # # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if training_args.local_rank == 0:
        print("YES")
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()
