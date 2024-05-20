import random
import re
from tqdm import tqdm
from os import listdir
import openai
import os
import pandas as pd
import time
from pathlib import Path
import random
import re
from tqdm import tqdm

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../prompt")


def get_prompt(name):
    with open(os.path.join(PROMPT_DIR, name + ".txt")) as f:
        return "".join([line for line in f])


def get_completion(model, prompt, max_tokens, temperature=0):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].text