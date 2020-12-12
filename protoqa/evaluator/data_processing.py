import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import *

from utils import query_yes_no

try:
    import pandas as pd
    import xlrd

    ABLE_TO_LOAD_EXCEL = True
except ModuleNotFoundError:
    ABLE_TO_LOAD_EXCEL = False

__all__ = [
    "ABLE_TO_LOAD_EXCEL",
    "default_string_preprocessing",
    "load_data_from_excel",
    "load_data_from_jsonl",
    "save_to_jsonl",
    "save_question_cluster_data_to_input_jsonl",
    "load_predictions_from_jsonl",
    "load_ranking_data",
    "convert_ranking_data_to_answers",
]


def default_string_preprocessing(pred_answer: str, length_limit: int = 50) -> str:
    return pred_answer.lower()[:length_limit].strip()


if ABLE_TO_LOAD_EXCEL:

    def _load_excel_sheets(
        data_path: Union[Path, str]
    ) -> Tuple[Dict[str, pd.DataFrame], str]:
        """
        Loads excel data with multiple sheets.
        :param data_path:  Path to excel file.
        :return: Sheets, which is a Dict of sheet names to pandas DataFrames, and a hash of the data
        """
        if not ABLE_TO_LOAD_EXCEL:
            raise Exception(
                "Was not able to import pandas and xlrd, which is required for conversion."
            )
        else:
            data_path = Path(data_path)
            with open(data_path, mode="rb") as f:
                data_hash = hashlib.md5(f.read()).hexdigest()
                sheets = pd.read_excel(f, sheet_name=None)
            return sheets, data_hash


def load_data_from_jsonl(data_path: Union[Path, str]) -> Dict:
    """
    Load jsonl input data, converting the answers-cleaned item to use frozenset keys.
    :param data_path: path to jsonl data
    :return: question_id indexed Dict
    """
    question_data = dict()
    with open(data_path) as data:
        for q in data:
            q_json = json.loads(q)
            if isinstance(q_json["answers-cleaned"], list):
                q_json["answers-cleaned"] = {
                    frozenset(ans_cluster["answers"]): ans_cluster["count"]
                    for ans_cluster in q_json["answers-cleaned"]
                }
            # The following attempts to handle various formats of the data
            if "normalized-question" not in q_json:
                q_json["normalized-question"] = q_json["question"][
                    "normalized-question"
                ]
            if "metadata" in q_json:
                question_data[q_json["metadata"]["id"]] = q_json
            else:
                question_data[q_json["questionid"]] = q_json
    return question_data


def load_data_from_excel(data_path: Union[Path, str], round: int = 1) -> Dict:
    """
    Loads data from the clustering excel files.
    :param data_path: Path to the excel file.
    :param round: Clustering round, only used for creating the question id
    :return: Dict with question_id to all data for this question.
    """
    sheets, data_hash = _load_excel_sheets(data_path)
    question_data = dict()
    for sheet_idx, (sheet_name, sheet) in enumerate(sheets.items()):
        # only work with the numbered sheets
        try:
            int(sheet_name)
        except:
            continue

        q_dict = dict()
        sheet = sheet.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        fixed_spelling = sheet.columns[2]
        combined = sheet.columns[5]

        for expected_value, actual_value in {
            "fixed_spelling": fixed_spelling,
            "combined": combined,
        }.items():
            if actual_value.lower().replace(" ", "_") != expected_value:
                warnings.warn(
                    f"Expected column named {expected_value}, got {actual_value} in sheet = {sheet_name}, file = {data_path.name}"
                )
                if not query_yes_no("Do you want to continue anyway?"):
                    raise ValueError("File {data_path.name} was malformed.")

        q_dict["raw-original-answers"] = Counter(sheet["answer"].dropna().astype(str))
        q_dict["raw-answers-cleaned"] = Counter(
            sheet[fixed_spelling].dropna().astype(str)
        )
        clusters = (
            sheet.loc[
                sheet[fixed_spelling].notna() & (sheet[combined] != "?"),
                [fixed_spelling, combined],
            ]
            .astype(str)
            .groupby(combined)[fixed_spelling]
            .agg(count=len, frozenset=frozenset)
        )
        q_dict["answers-cleaned"] = {
            row["frozenset"]: row["count"] for _, row in clusters.iterrows()
        }
        questionid = f"r{round}q{sheet_name}"

        q_dict["ann1"] = list(sheet.iloc[:, 3])
        q_dict["ann2"] = list(sheet.iloc[:, 4])

        question_data[questionid] = {
            "question": sheet["question"][0],
            "do-not-use": isinstance(sheet["question"][1], str)
            and sheet["question"][1].replace(" ", "").lower() == "donotuse",
            "normalized-question": sheet["question"][0].lower(),
            "source": data_path.name,
            "source-md5": data_hash,
            "sourceid": sheet_name,
            "questionid": questionid,
            **q_dict,
        }
    return question_data


def save_to_jsonl(data_path: Union[Path, str], qa_dict: Dict) -> None:
    """
    Saves answer data from the load_data_from_excel format, which uses frozensets for answer keys, into a jsonl format.
    """
    with open(data_path, "w") as output_file:
        for qa in qa_dict.values():
            qa = qa.copy()
            if "answers-cleaned" in qa:
                qa["answers-cleaned"] = [
                    {"count": count, "answers": list(answers)}
                    for answers, count in qa["answers-cleaned"].items()
                ]
            json.dump(qa, output_file)
            output_file.write("\n")


def save_question_cluster_data_to_input_jsonl(
    data_path: Union[Path, str], q_dict: Dict
) -> None:
    """
    Creates a stub file which contains the question_id, question, and an empty placeholder list for ranked_answers
    """
    with open(data_path, "w") as output_file:
        for qid, q in q_dict.items():
            q_new = {}
            q_new["question_id"] = qid
            q_new["question"] = q["question"]
            q_new["ranked_answers"] = []
            json.dump(q_new, output_file)
            output_file.write("\n")


def load_predictions_from_jsonl(data_path: Union[Path, str]) -> Dict:
    """
    Loads jsonl into a simplified dictionary structure which only maps qids to lists of answers.
    """

    if str(data_path).endswith("jsonl"):
        ans_dict = dict()
        fin = open(data_path)
        for line in fin:
            print(line)
            line = json.loads(line.strip())
            if "question_id" in line:
                qid = line["question_id"]
                ans = line["ranked_answers"]
                ans_dict[qid] = ans
            else:
                ans_dict.update(line)
        fin.close()
        return ans_dict
    else:
        return json.load(open(data_path))


def load_ranking_data(data_path: Union[Path, str]) -> Dict[str, List[str]]:
    """
    Load in the ranking data from the human ranking evaluation task.
    :param data_path:
    :return:
    """
    sheets, data_hash = _load_excel_sheets(data_path)
    all_answers = dict()
    for sheet_idx, (sheet_name, sheet) in enumerate(sheets.items()):
        # only work with the numbered sheets
        try:
            int(sheet_name)
        except:
            continue
        answers = sheet.iloc[0:5].transpose()
        answers.columns = answers.iloc[
            0
        ]  # use the questions themselves as column headers
        answers = answers.drop(
            answers.index[0:2]
        )  # drop the questions and "completed?" rows
        answers_dict = {
            k: [x for x in v if pd.notnull(x)]
            for k, v in answers.to_dict("list").items()
        }
        assert set(all_answers.keys()).intersection(answers_dict.keys()) == set()
        all_answers.update(answers_dict)
    return all_answers


def convert_ranking_data_to_answers(
    ranking_data: Dict[str, List[str]],
    question_data: Dict,
    allow_incomplete: bool = False,
) -> Dict[str, Dict[str, Union[str, List[str]]]]:
    question_to_ids = {v["question"]: k for k, v in question_data.items()}

    if not allow_incomplete:
        completed_rankings = {k: v for k, v in ranking_data.items() if len(v) >= 10}
        if len(completed_rankings) < len(ranking_data):
            warnings.warn(
                f"Missing completed rankings for {len(ranking_data)-len(completed_rankings)} questions."
            )
    else:
        completed_rankings = ranking_data

    questions_in_both = set(question_to_ids.keys()).intersection(
        completed_rankings.keys()
    )
    if len(questions_in_both) < len(completed_rankings):
        warnings.warn(
            f"Missing ground-truth clusters for {len(completed_rankings)-len(questions_in_both)} completed rankings."
        )
        print(set(completed_rankings.keys()).difference(question_to_ids.keys()))

    answers_dict = {
        question_to_ids[k]: {
            "question": k,
            "predicted_answer": ranking_data[k],
            "question_id": question_to_ids[k],
        }
        for k in questions_in_both
    }
    return answers_dict
