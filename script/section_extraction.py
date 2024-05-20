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
from collections import OrderedDict

input_sections = OrderedDict([
    ('Allergies', 'Allergies'),
    ('Chief Complaint', '(?:Chief|_+) Complaint'),
    ('Major Surgical or Invasive Procedure', '(?:Major |_+ *)(?:Surgical |_+ *)(?:or |_+ *)(?:Invasive|_+ *) Procedure'),
    ('History of Present Illness', '(?:History|_+) of Present Illness'),
    ('Past Medical History', '(?:Past|_+) Medical History'),
    ('Social History', '(?:Social|_+) History'),
    ('Family History', '(?:Family|_+) History'),
    ('Physical Exam', 'Physical [A-Za-z_]+'),
    ('Pertinent Results', '(?:Pertinent|_+) Results'),
    ('Brief Hospital Course', 'Brief Hospital Course'),
    ('Medications on Admission', '[A-Za-z_]+ on Admission'),
    ('Discharge Medications', '[A-Za-z_]+ Medications'),
    ('Discharge Disposition', '[A-Za-z_]+ Disposition'),
    ('Discharge Diagnosis', '[A-Za-z_]+ Diagnosis'),
    ('Discharge Condition', '[A-Za-z_]+ Condition'),
    ('Discharge Instructions', 'Discharge Instructions')
])


def parse_sections(row):
    discharge_summary = row['text']

    for i, (section_name, section) in enumerate(input_sections.items()):
        if section_name in ['Pertinent Results', 'Physical Exam', 'Brief Hospital Course', 'Past Medical History', 'Social History', 'Family History']:
            for next_section in list(input_sections.values())[ i +1:]:
                search = re.findall(section + ".+\n" + next_section, discharge_summary, re.DOTALL)
                if len(search) > 0:
                    break
            rex = r'(%s?):\s*\n{0,2}(.+?)\s*(\n\s*){1,10}(%s):\n' % (section, next_section)
        else:
            rex = r'(%s?):\s*\n{0,2}(.+?)\s*(\n\s*){1,10}((?:[A-Z][a-z ]+)+):' % (section)

        section_ext = re.findall(rex, discharge_summary, re.DOTALL)
        section_col_name = section_name.replace(" ", "_")
        if len(section_ext) > 0:
            row[section_col_name] = section_ext[-1]
        else:
            row[section_col_name] = np.nan

    return row


def remove_output_from_input(row):
    if 'brief_hospital_course' in row:
        # Use Generated Brief Hospital Course for subsequent generation of Discharge Instructions
        # if 'Brief Hospital Course' in row['text']:
        if pd.notnull(row['Brief_Hospital_Course']):
            row['new_text'] = row['text'].replace(row['Brief_Hospital_Course'], row['brief_hospital_course'])
        else:
            row['new_text'] = row['text'] + "\n\nBrief Hospital Course:\n" + row['brief_hospital_course']
    else:
        row['new_text'] = row['text'].replace(row['Brief_Hospital_Course'], '')
        row['new_text'] = re.sub(r'Brief Hospital Course:\n*', r'', row['new_text'], flags=re.DOTALL)

    row['new_text'] = row['new_text'].replace(row['Discharge_Instructions'], '')
    row['new_text'] = re.sub(r'Discharge Instructions:\n*', r'', row['new_text'], flags=re.DOTALL)

    row['new_text'] = re.sub(r'(\n ?)+(Followup Instructions)', r'\n\n\n\2', row['new_text'], flags=re.DOTALL)

    return row


field = ['Brief_Hospital_Course', 'Discharge_Instructions', 'Physical_Exam', 'Pertinent_Results']


def calculate_word_count(row):
    discharge_sections = []
    for col in field:
        word_count = 0
        if pd.notnull(row[col]):
            word_count = len(row[col].split(" "))
        else:
            word_count = 0
        row[col + "_Word_Count"] = word_count

    return row

