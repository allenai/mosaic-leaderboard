import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import *

from utils import query_yes_no

try:
    import pandas as pd
    import openpyxl

    ABLE_TO_LOAD_EXCEL = True
except ModuleNotFoundError:
    ABLE_TO_LOAD_EXCEL = False

__all__ = [
    "ABLE_TO_LOAD_EXCEL",
    "QuestionAndAnswerClusters",
    "default_string_preprocessing",
    "load_data_from_excel",
    "load_question_answer_clusters_from_jsonl",
    "load_crowdsourced_xlsx_to_predictions",
    "save_to_jsonl",
    "load_predictions_from_jsonl",
    "save_predictions_to_jsonl",
    "load_ranking_data",
    "convert_ranking_data_to_answers",
    "convert_scraped_old_to_new",
    "convert_crowdsourced_old_to_new",
    "convert_old_list_to_new",
    "load_jsonl_to_list",
    "save_list_to_jsonl",
]


class QuestionAndAnswerClusters(NamedTuple):
    question_id: str
    question: str
    answer_clusters: Dict[frozenset, int]


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
                "Was not able to import pandas and openpyxl, which is required for loading Excel files."
            )
        else:
            data_path = Path(data_path)
            with data_path.open("rb") as f:
                data_hash = hashlib.md5(f.read()).hexdigest()
                sheets = pd.read_excel(f, sheet_name=None, engine="openpyxl")
            return sheets, data_hash


else:

    def _load_excel_sheets(*args, **kwargs):
        raise ValueError(
            "You must install pandas and openpyxl in order to read Excel files."
        )


def _check_column_name(
    actual_value: str, expected_value: str, sheet_name: str, file_name: str
) -> None:
    if actual_value.lower().replace(" ", "_") != expected_value:
        warnings.warn(
            f"Expected column named {expected_value}, got {actual_value} in sheet = {sheet_name}, file = {file_name}"
        )
        if not query_yes_no("Do you want to continue anyway?"):
            raise ValueError("File {file_name} was malformed.")


def load_question_answer_clusters_from_jsonl(
    data_path: Union[Path, str]
) -> Dict[str, QuestionAndAnswerClusters]:
    """
    Load jsonl answer cluster data for evaluation.
    Attempts to handle multiple formats of input jsonl files.

    :param data_path: path to jsonl data
    :return: Dict from question_id to QuestionAndAnswerClusters
    """
    question_data = dict()
    with open(data_path) as data:
        for q in data:
            q_json = json.loads(q)
            if "answers" in q_json and "clusters" in q_json["answers"]:
                answer_clusters = {
                    frozenset(answer_cluster["answers"]): answer_cluster["count"]
                    for answer_cluster in q_json["answers"]["clusters"].values()
                }
            else:
                warnings.warn(
                    f"Data in {data_path} seems to be using an old format. "
                    f"We will attempt to load anyway, but you should download the newest version."
                )
                if "answers-cleaned" in q_json and isinstance(
                    q_json["answers-cleaned"], list
                ):
                    answer_clusters = {
                        frozenset(ans_cluster["answers"]): ans_cluster["count"]
                        for ans_cluster in q_json["answers-cleaned"]
                    }
                else:
                    raise ValueError(f"Could not load data from {data_path}.")

            if "question" in q_json and "normalized" in q_json["question"]:
                question = q_json["question"]["normalized"]
            else:
                warnings.warn(
                    f"Data in {data_path} seems to be using an old format. "
                    f"We will attempt to load anyway, but you should download the newest version."
                )
                if "normalized-question" in q_json:
                    question = q_json["normalized-question"]
                elif (
                    "question" in q_json and "normalized-question" in q_json["question"]
                ):
                    question = q_json["question"]["normalized-question"]
                else:
                    raise ValueError(f"Could not load data from {data_path}.")

            if "metadata" in q_json and "id" in q_json["metadata"]:
                question_id = q_json["metadata"]["id"]
            else:
                warnings.warn(
                    f"Data in {data_path} seems to be using an old format. "
                    f"We will attempt to load anyway, but you should download the newest version."
                )
                if "questionid" in q_json:
                    question_id = q_json["questionid"]
                else:
                    raise ValueError(f"Could not load data from {data_path}.")

            question_data[question_id] = QuestionAndAnswerClusters(
                question_id=question_id,
                question=question,
                answer_clusters=answer_clusters,
            )
    return question_data


def load_data_from_excel(
    data_path: Union[Path, str], round: int = 1
) -> Dict[str, Dict]:
    """
    Loads data from the clustering excel files.

    :param data_path: Path to the excel file
    :param round: Clustering round, only used for creating the question id
    :return: Dict with question_id to all data for this question in dictionary format
    """
    data_path = Path(data_path).expanduser()
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
            _check_column_name(expected_value, actual_value, sheet_name, data_path.name)

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


def save_to_jsonl(data_path: Union[Path, str], qa_dict: Dict[str, Dict]) -> None:
    """
    Saves answer data from the load_data_from_excel format, which uses frozensets for answer keys, into a jsonl format.

    :param data_path: location to save question answer dictionary as jsonl
    :param qa_dict: dictionary of question and answer data, as from the output of load_data_from_excel
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


def load_predictions_from_jsonl(data_path: Union[Path, str]) -> Dict[str, List[str]]:
    """
    Loads jsonl into a simplified dictionary structure which only maps qids to lists of answers.
    Each line in the jsonl file should have the following format:
    ```
    {
      "question_id": <str> question id,
      "ranked_answers": ["answer_one", "answer_two", ...]
    }
    If data_path is a json file instead, it should already be a map from question id to a ranked list of answers, eg.
    {
        "r1q1": ["answer_one", "answer_two", ...],
        "r1q2": ["answer_one", "answer_two", ...],
        ...
    }
    ```
    :param data_path: Path for jsonl file
    :return: Dict from question id to ranked list of answer strings
    """
    data_path = Path(data_path).expanduser()

    if str(data_path).endswith("jsonl"):
        ans_dict = dict()
        fin = open(data_path)
        for line in fin:
            line = json.loads(line.strip())
            if "question_id" in line:
                qid = line["question_id"]
                ans = line["ranked_answers"]
                ans_dict[qid] = ans
            else:
                warnings.warn(
                    f"Predictions file {data_path} does not seem to be in the correct format. "
                    f"We will continue to load it, but you may experience errors in evaluation."
                )
                ans_dict.update(line)
        fin.close()
        return ans_dict
    else:
        return json.load(open(data_path))


def load_crowdsourced_xlsx_to_predictions(
    data_path: Union[Path, str], questions_to_ids: Dict[str, str]
) -> Dict[str, List[str]]:
    """
    Load in the crowdsourced "fixed spelling" data from the human ranking evaluation task.
    Excel file should have the following format:
    A1 = Question string
    B = answers column (with "answers" as header in B1)
    C = fixed_spelling column (with "fixed_spelling" as header in C1)

    :param data_path: location of xlsx file
    :param questions_to_ids: dictionary mapping question strings to ids
    :return: predictions dictionary
    """
    data_path = Path(data_path).expanduser()
    sheets, _ = _load_excel_sheets(data_path)
    predictions = dict()
    for sheet_idx, (sheet_name, sheet) in enumerate(sheets.items()):
        sheet = sheet.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        fixed_spelling = sheet.columns[2]
        _check_column_name(fixed_spelling, "fixed_spelling", sheet_name, data_path.name)

        answer_counts = Counter(sheet[fixed_spelling].dropna().astype(str))
        ranked_answers = [
            answer_string for answer_string, _ in answer_counts.most_common()
        ]
        question_id = questions_to_ids[sheet.columns[0].strip().lower()]
        predictions[question_id] = ranked_answers

    return predictions


def save_predictions_to_jsonl(
    predictions: Dict[str, List[str]], file_path: Union[Path, str]
) -> None:
    """
    Save a predictions dictionary to jsonl.

    :param predictions: dict from question ids to ranked list of answers
    :param file_path: location to save the data
    """
    file_path = Path(file_path)
    if file_path.suffix == "json":
        with file_path.open("w") as f:
            json.dump(predictions, f)
    else:
        save_list_to_jsonl(
            [
                {"question_id": question_id, "ranked_answers": ranked_answers}
                for question_id, ranked_answers in predictions.items()
            ],
            file_path,
        )


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
    question_data: List[Dict[str, Any]],
    allow_incomplete: bool = False,
) -> List[Dict[str, List[str]]]:
    question_to_ids = {q["question"]: q["questionid"] for q in question_data}

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

    answers_list = [{question_to_ids[k]: ranking_data[k]} for k in questions_in_both]
    return answers_list


def convert_scraped_old_to_new(old: Dict[str, Any]) -> Dict[str, Any]:
    """Convert scraped data from the original format (which had "answer-clusters" and "answerstrings" fields) to a more succinct format."""
    raw = {
        cluster["answers"][0]: cluster["count"] for cluster in old["answer-clusters"]
    }
    clusters = {
        cluster["clusterid"]: {"count": cluster["count"], "answers": cluster["answers"]}
        for cluster in old["answer-clusters"]
    }
    num_answers = sum([cluster["count"] for cluster in clusters.values()])
    assert num_answers == old["metadata"]["totalcount"]
    assert len(raw) == len(old["answerstrings"])
    assert set(old["answerstrings"].keys()) == set(raw.keys())
    new = {
        "metadata": {
            "id": old["metadata"]["id"],
            "source": old["metadata"]["source"],
        },
        "question": {
            "original": old["question"]["question"],
            "normalized": old["question"]["normalized-question"],
        },
        "answers": {
            "raw": raw,
            "clusters": clusters,
        },
        "num": {"answers": num_answers, "clusters": len(clusters)},
    }
    return new


def convert_crowdsourced_old_to_new(old: Dict[str, Any]) -> Dict[str, Any]:
    """Convert scraped data from the original format (which had "answers-cleaned" field) to a clearer format."""
    raw = old["answerstrings"]
    clusters = {
        cluster["clusterid"]: {"count": cluster["count"], "answers": cluster["answers"]}
        for cluster in old["answers-cleaned"]
    }
    num_answers = sum([cluster["count"] for cluster in clusters.values()])
    assert num_answers == old["metadata"]["totalcount"]
    new = {
        "metadata": {
            "id": old["metadata"]["id"],
            "source": old["metadata"]["source"],
        },
        "question": {
            "original": old["question"]["question"],
            "normalized": old["question"]["normalized-question"],
        },
        "answers": {
            "raw": raw,
            "clusters": clusters,
        },
        "num": {"answers": num_answers, "clusters": len(clusters)},
    }
    return new


def convert_drive_files_old_to_new(old: Dict[str, Any]) -> Dict[str, Any]:
    """Convert data from the original format (which had "answers-cleaned" field) to a clearer format."""
    question_id = old["questionid"]
    raw = old["raw-original-answers"]
    clusters = {
        f"{question_id}.{cluster_num}": {
            "count": cluster["count"],
            "answers": cluster["answers"],
        }
        for cluster_num, cluster in enumerate(old["answers-cleaned"])
    }
    num_answers = sum([cluster["count"] for cluster in clusters.values()])
    new = {
        "metadata": {
            "id": question_id,
            "source": "umass-crowdsource",
        },
        "question": {
            "original": old["question"],
            "normalized": old["normalized-question"],
        },
        "answers": {
            "raw": raw,
            "clusters": clusters,
        },
        "num": {"answers": num_answers, "clusters": len(clusters)},
    }
    return new


def convert_old_list_to_new(old: List) -> List:
    if "answers-cleaned" in old[0]:
        if "answerstrings" in old[0]:
            return [convert_crowdsourced_old_to_new(x) for x in old]
        else:
            return [convert_drive_files_old_to_new(x) for x in old]
    else:
        return [convert_scraped_old_to_new(x) for x in old]


def load_jsonl_to_list(file_path: Union[Path, str]) -> List:
    file_path = Path(file_path).expanduser()
    output = []
    with file_path.open() as f:
        for l in f:
            output.append(json.loads(l.strip()))
    return output


def save_list_to_jsonl(list_data: List, file_path: Union[Path, str]) -> None:
    file_path = Path(file_path).expanduser()
    with file_path.open("w") as output_file:
        for l in list_data:
            json.dump(l, output_file)
            output_file.write("\n")
