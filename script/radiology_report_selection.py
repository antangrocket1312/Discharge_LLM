import sys
import argparse
import os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils.prompting import *
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


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
        new_pertinent_results += "=============\n\n"

        row['text'] = row['text'].replace(row['Pertinent_Results'], new_pertinent_results)

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='sample',
                        help="The directory containing discharge summary and radiology input")
    parser.add_argument("--section_extracted_input", type=str, default='discharge_extracted.pkl',
                        help="The name of the discharge summary file with all sections extracted")
    parser.add_argument("--output_file_name", type=str, default='discharge_processed.pkl',
                        help="The name of the output file to be saved")

    args = parser.parse_args()

    dataset = args.dataset
    data_path = f"./data/{dataset}/"

    # Load Dataset
    df = pd.read_pickle(os.path.join(data_path, args.section_extracted_input))

    # Radiology Report Selection
    df = df.parallel_apply(replace_pertinent_results_with_radiology, axis=1)

    # Export file
    df.to_pickle(os.path.join(data_path, args.output_file_name))
