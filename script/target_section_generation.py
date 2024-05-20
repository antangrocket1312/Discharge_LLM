import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils.prompting import *
from datasets import concatenate_datasets, load_dataset
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import torch
import os
import ast
import spacy
from utils.prompting import *
# pd.set_option('display.max_colwidth', None)
import time
from multiprocessing import Pool
from tqdm import tqdm
from os import listdir
import openai
import os
import warnings
warnings.filterwarnings("ignore")

def generate_target_section(target_section, data):
    if target_section == 'brief_hospital_course':
        my_prompt = "<s>[INST] " + get_prompt("brief_hospital_course") + "\n\nDischarge summary:\"\"\"%s\"\"\" [/INST]"
        model = 'Mistral-7B-Instruct-v0.2-GPTQ-Brief-Hospital-Course'
        max_length = 1000
    elif target_section == 'discharge_instructions':
        my_prompt = "<s>[INST] " + get_prompt("discharge_instructions") + "\n\nDischarge summary:\"\"\"%s\"\"\" [/INST]"
        model = 'Mistral-7B-Instruct-v0.2-GPTQ-Discharge_Instructions'
        max_length = 500

    final_prompt = my_prompt % data
    retries = 5
    while retries > 0:
        try:
            response = get_completion(model, final_prompt, max_length, temperature=0)
            return response
        except Exception as e:
            if e:
                if "exceeded your current quota" in str(e).lower():
                    raise e
                print(e)
                print('Timeout error, retrying...')
                retries -= 1
                if "limit reached for" in str(e).lower():
                    time.sleep(30)
                else:
                    time.sleep(5)
            else:
                raise e

    print('API is not responding, moving on...')
    return None


def target_section_summarization(root_path, target_section, domain, domain_df, save_step=10):
    if target_section not in ['brief_hospital_course', 'discharge_instructions']:
        raise ValueError('Invalid Target Section. Must either be: \'brief_hospital_course\' or '
                         '\'discharge_instructions\'')

    src_path = f"{root_path}/{domain}"
    Path(src_path).mkdir(parents=True, exist_ok=True)
    extractions = []

    file_names = listdir(src_path)
    postfix = [re.split("[_.]", name)[1]
               for name in listdir(src_path)
               ]
    start = 0
    if 'done' in postfix:
        print(domain, ": ", "Loaded cached file. Done")
        new_domain_df = pd.read_pickle(f"{src_path}/{domain}_done.pkl")
        return new_domain_df
    elif len(postfix) > 0:
        last_index = max([int(idx) for idx in postfix if idx != 'done'])
        last_domain_df = pd.read_pickle(f"{src_path}/{domain}_{last_index}.pkl")
        extractions = last_domain_df[target_section].tolist()
        start = last_index
        print(domain, "Loaded saved file. Continuing")
    else:
        print(domain, "Start new process.")

    for i, (_, row) in tqdm(enumerate(domain_df.iterrows()), total=domain_df.shape[0]):
        if i < start:
            continue

        discharge_summary_data = row['processed_text']
        extraction = generate_target_section(target_section, discharge_summary_data)
        time.sleep(0.3)
        extractions += [extraction]

        if (i + 1) % save_step == 0:
            save_df = domain_df.iloc[:i + 1]
            save_df.insert(0, target_section, extractions)
            save_df[['hadm_id', target_section]].to_pickle(f"{src_path}/{domain}_{i + 1}.pkl")

    new_domain_df = domain_df.iloc[:i + 1]
    new_domain_df.insert(0, target_section, extractions)
    new_domain_df[['hadm_id', target_section]].to_pickle(f"{src_path}/{domain}_done.pkl")
    return new_domain_df


def remove_repitition(text):
    truncated_text = []
    prev_sent = None
    for sent in text.split("\n"):
        if (sent == '' and prev_sent != '') or (sent not in truncated_text):
            prev_sent = sent
            truncated_text += [sent]

    return "\n".join(truncated_text)


def remove_sent_repitition(text):
    truncated_text = []
    prev_sent = None
    for sent in text.split(". "):
        if (sent == '' and prev_sent != '') or (sent not in truncated_text):
            prev_sent = sent
            truncated_text += [sent]

    return ". ".join(truncated_text)


def fix_hallucination(row):
    if pd.notnull(row['History_of_Present_Illness']):
        text = row['History_of_Present_Illness']
    else:
        text = row['brief_hospital_course']
    return text


def post_process_brief_hospital_course(row):
    text = row['brief_hospital_course']
    text = re.sub(r'^(?i)\n*brief hospital course:*\n*', r'', text, re.DOTALL)

    text = re.sub(r'\n*(.*?)(?i)medications on admission:*(\n*[\s ]*\n*)*([^\n]+\n)+(\n*[^\n]*[A-Z])', r'\n\4', text,
                  re.DOTALL)
    # RM Discharge medications or follow up parts like Discharge disposition, Facility. Discharge diagnosis
    text = re.sub(r'\n*(.*?)(?i)discharge medications:*(\n*[\s ]*\n*)*([^\n]+\n*)+$', r'', text, re.DOTALL)
    text = re.sub(r'\n*(.*?)(?i)medications on discharge::*(\n*[\s ]*\n*)*([^\n]+\n*)+$', r'', text, re.DOTALL)

    return text


def post_process_discharge_instructions(row):
    text = row['discharge_instructions']

    text = re.sub(r'^(?i)\n*discharge instructions:*\n*', r'', text, re.DOTALL)

    text_splits = text.split("\n")
    if "pleasure taking care" not in text:
        thanks = 'It was a pleasure taking care of you at ___.'
        text_splits = [text_splits[0]] + [thanks] + text_splits[1:]

    if len(re.findall("(?i)(sincerely,[\n ]+your.+team.*)", text)) == 0:
        greet = "Sincerely,\nYour ___ Care Team"
        text_splits += [greet]

    text = "\n".join(text_splits)
    text = re.sub(r'(?i)(sincerely,[\n ]+your.+team.*)[\n ]+([^\n]+\n*)+$', r'\1', text, re.DOTALL)

    return text
