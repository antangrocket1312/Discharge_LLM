import sys
import numpy as np
import argparse
import os
import warnings
from collections import OrderedDict
from pandarallel import pandarallel

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils.prompting import *

warnings.filterwarnings("ignore")

input_sections = OrderedDict([
    ('Allergies', 'Allergies'),
    ('Chief Complaint', '(?:Chief|_+) Complaint'),
    (
        'Major Surgical or Invasive Procedure',
        '(?:Major |_+ *)(?:Surgical |_+ *)(?:or |_+ *)(?:Invasive|_+ *) Procedure'),
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
pandarallel.initialize(progress_bar=True)


def merge_two_dfs_no_dup_cols(df1, df2, merge_col):
    cols_to_use = df2.columns.difference(df1.columns).tolist() + merge_col
    return df1.merge(df2[cols_to_use], on=merge_col)


def parse_sections(row):
    discharge_summary = row['text']

    for i, (section_name, section) in enumerate(input_sections.items()):
        if section_name in ['Pertinent Results', 'Physical Exam', 'Brief Hospital Course', 'Past Medical History',
                            'Social History', 'Family History']:
            for next_section in list(input_sections.values())[i + 1:]:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='sample',
                        help="The directory containing discharge summary and radiology input")
    parser.add_argument("--output_file_name", type=str, default='discharge_extracted.pkl',
                        help="The name of the output file to be saved")
    args = parser.parse_args()

    dataset = args.dataset
    data_path = f"./data/{dataset}/"

    # Load Dataset
    dfs = {}
    df_discharge = pd.read_csv(os.path.join(data_path, "discharge.csv.gz"), keep_default_na=False)
    df_radiology = pd.read_csv(os.path.join(data_path, "radiology.csv.gz"), keep_default_na=False)
    merged_df = merge_two_dfs_no_dup_cols(df_discharge, df_radiology.rename(columns={'text': 'radiology_text'}),
                                          ['hadm_id'])
    merge_col = merged_df.drop(columns=['radiology_text']).columns.tolist()
    merged_df = merged_df.groupby(merge_col).agg({
        'radiology_text': lambda x: x.tolist()
    }).reset_index()
    df = merged_df

    # Parse Section
    df = df.parallel_apply(parse_sections, axis=1)
    sections_col = [col.replace(" ", "_") for col in input_sections.keys()]
    for col in sections_col:
        mask = pd.notnull(df[col])
        df.loc[mask, col] = df.loc[mask, col].apply(lambda x: x[1])

    # Export file
    df.to_pickle(os.path.join(data_path, args.output_file_name))
