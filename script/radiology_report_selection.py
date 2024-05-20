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


def replace_pertinent_results_with_radiology(row):
    if pd.notnull(row['Pertinent_Results']):
        new_reports = []
        for report in row['radiology_text']:
            rex = r'((?i)impression:[\s ]*\n{0,2}(.+?)\s*$)'
            section_ext = re.findall(rex, report, re.DOTALL)
            if len(section_ext) > 0 and section_ext[0][1][:15] in row['Pertinent_Results']:
                new_reports += [report]

        new_pertinent_results = ""
        #         new_reports.sort(key=len, reverse=True)
        for report in new_reports:
            if len(new_pertinent_results.split(" ")) == 1:
                new_pertinent_results += report
            elif len(new_pertinent_results.split(" ")) < 1000:
                new_pertinent_results += ("=============\n\n" + report)
        #         new_pertinent_results = "=============\n\n".join([report for report in new_reports])
        new_pertinent_results += "=============\n\n"

        row['processed_text'] = row['processed_text'].replace(row['Pertinent_Results'], new_pertinent_results)

    return row
