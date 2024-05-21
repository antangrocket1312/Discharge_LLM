import os
import argparse
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils.prompting import *
import time
from multiprocessing import Pool
from tqdm import tqdm
from os import listdir
import openai
import os
import warnings
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='sample',
                        help="The directory containing discharge summary and radiology input")
    parser.add_argument("--target_section", type=str, required=True,
                        choices={"brief_hospital_course", "discharge_instructions"},
                        help="The target section of the discharge summary to be generated. Must either be "
                             "\"brief_hospital_course\" or \"discharge_instructions\"")
    parser.add_argument("--processed_input", type=str, default='discharge_processed.pkl',
                        help="The name of the discharge summary file with all sections extracted and selected for "
                             "good radiology information")
    parser.add_argument("--num_workers", type=int, default=3, help="The number of workers for the ABSA task")

    args = parser.parse_args()

    dataset = args.dataset
    data_path = f"./data/{dataset}/"

    target_section = args.target_section
    num_workers = args.num_workers

    # Load Dataset
    df = pd.read_pickle(os.path.join(data_path, args.processed_input))

    if target_section == 'discharge_instructions':
        if os.path.isfile(f"./data/{dataset}/brief_hospital_course.csv"):
            # Read generated Brief Hospital Course
            brief_hospital_course_df = pd.read_csv(f"./data/{dataset}/brief_hospital_course.csv")
            df = df.merge(brief_hospital_course_df, on=['hadm_id'])
        else:
            raise FileNotFoundError("\"brief_hospital_course\" must be generated before generating discharge "
                                    "instructions")

    # Pre-processing
    df = df.apply(remove_output_from_input, axis=1)
    df = df.rename(columns={'new_text': 'processed_text'})
    df = df.apply(calculate_word_count, axis=1)
    df['processed_text_word_count'] = df['processed_text'].apply(lambda x: len(x.split(" ")))
    df = df.sort_values(by=['processed_text_word_count'], ascending=False)

    # Partition
    thres = 1000
    df['category'] = df['processed_text_word_count'].apply(lambda x: 1 if x < thres else 0)
    mask = (df['processed_text_word_count'] >= 1000) & (df['processed_text_word_count'] <= 1300)
    df.loc[mask, 'category'] = 2

    root_path = f'./data/{dataset}/{target_section}_cache/'

    inputs = [(root_path,
               target_section,
               domain,
               df[df['category'] == domain].reset_index(drop=True),
               100,
               )
              for domain in [0, 1, 2]]
    start_time = time.time()
    with Pool(num_workers) as processor:
        data = processor.starmap(target_section_summarization, inputs)

    processed_df = pd.concat(data).drop_duplicates(['hadm_id'])
    df = df.merge(processed_df, on=['hadm_id'])

    # Post-processing
    if target_section == 'brief_hospital_course':
        mask = df['brief_hospital_course'].apply(lambda x: len(re.findall(r'(\#[^\n]+\n){8,}', x)) > 0)
        df.loc[mask, 'brief_hospital_course'] = df[mask].apply(fix_hallucination, axis=1)

        df['brief_hospital_course'] = df['brief_hospital_course'].apply(remove_repitition)
        df['brief_hospital_course'] = df['brief_hospital_course'].apply(lambda x: "\n\n".join(x.split("\n\n")[0:3]))
        df['brief_hospital_course'] = df['brief_hospital_course'].apply(
            lambda text: re.sub(r'([A-Za-z0-9,._][ \s])\n([A-Za-z0-9])', r'\1\2', text))
        df['brief_hospital_course'] = df['brief_hospital_course'].apply(remove_sent_repitition)

        mask = pd.notnull(df['brief_hospital_course'])
        df.loc[mask, 'brief_hospital_course'] = df[mask].parallel_apply(post_process_brief_hospital_course, axis=1)

        df = df.drop_duplicates(subset=['hadm_id'])

        df[['hadm_id', 'brief_hospital_course']].to_csv(f"./data/{dataset}/brief_hospital_course.csv", index=False)
    elif target_section == 'discharge_instructions':
        df['discharge_instructions'] = df['discharge_instructions'].apply(remove_repitition)
        df['discharge_instructions'] = df['discharge_instructions'].apply(
            lambda text: re.sub(r'([A-Za-z0-9,._][ \s])\n([A-Za-z0-9])', r'\1\2', text))
        df['discharge_instructions'] = df['discharge_instructions'].apply(remove_sent_repitition)

        mask = pd.notnull(df['discharge_instructions'])
        df.loc[mask, 'discharge_instructions'] = df[mask].parallel_apply(post_process_discharge_instructions, axis=1)

        df = df.drop_duplicates(subset=['hadm_id'])

        df[['hadm_id', 'discharge_instructions']].to_csv(f"./data/{dataset}/discharge_instructions.csv", index=False)

