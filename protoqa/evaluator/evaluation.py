from data_processing import QuestionAndAnswerClusters, default_string_preprocessing
from scoring import *
import statistics
import numpy as np
from functools import partial
from typing import *


def multiple_evals(
    eval_func_dict: Dict[str, Callable],
    question_data: Dict[str, QuestionAndAnswerClusters],
    answers_dict: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    eval_details = {}
    for name, eval_func in eval_func_dict.items():
        print(f"Evaluating {name}...", flush=True)
        eval_details[name] = evaluate(
            evaluation_func=eval_func,
            question_data=question_data,
            answers_dict=answers_dict,
        )
        eval_score = statistics.mean(x.score for x in eval_details[name].values())
        print(f"{name}: {eval_score}")
    return eval_details


def evaluate(
    evaluation_func: Callable,
    question_data: Dict[str, QuestionAndAnswerClusters],
    answers_dict: Dict[str, List[str]],
    data_preprocessing: Optional[Callable] = None,
) -> Dict[str, float]:
    scores = dict()
    for qid, pred_answers in answers_dict.items():
        true_q = question_data[qid]
        if data_preprocessing is not None:
            true_q, pred_answers = data_preprocessing(true_q, answers_dict)
        true_answers = true_q.answer_clusters.copy()
        scores[qid] = evaluation_func(
            pred_answers,
            true_answers,
            question_string=true_q.question,
        )
    return scores


class EvalResult(NamedTuple):
    score: float
    score_matrix: np.ndarray
    answer_assignment: dict

    def __eq__(self, other):
        return (
            self.score == other.score
            and (self.score_matrix == other.score_matrix).all()
            and self.answer_assignment == other.answer_assignment
        )


def general_eval(
    pred_answers,
    true_answers,
    *,
    max_pred_answers: Optional[int] = None,
    max_incorrect: Optional[int] = None,
    string_preprocessing: Callable = default_string_preprocessing,
    question_string: str = "question string",
    score_func: Callable = exact_match,
    cluster_score_func: Callable = cluster_score,
    cluster_reduction_func: Callable = np.max,
    score_matrix_transformation: Optional[Callable] = None,
    assign_cluster_scores: bool = True,
    calc_oracle_score: bool = True
) -> EvalResult:
    if max_pred_answers is not None:
        pred_answers = pred_answers[:max_pred_answers]
    else:
        if max_incorrect is not None:
            max_answers = len(true_answers) + max_incorrect
            pred_answers = pred_answers[:max_answers]
    pred_answers = [string_preprocessing(pred_answer) for pred_answer in pred_answers]
    score_matrix = cluster_score_func(
        pred_answers,
        true_answers,
        question_string=question_string,
        score_func=score_func,
        cluster_reduction_func=cluster_reduction_func
    )
    if score_matrix_transformation is not None:
        score_matrix = score_matrix_transformation(score_matrix)
    if max_incorrect is not None:
        score_matrix = limit_total_wrong(score_matrix, max_incorrect)
    if assign_cluster_scores:
        score_matrix *= np.array(list(true_answers.values()))[None]
    score, row_ind, col_ind = get_optimal_score(score_matrix)
    answer_assignment = dict()
    true_answers_list = list(true_answers.keys())
    for r, c in zip(row_ind, col_ind):
        answer_assignment[pred_answers[r]] = (
            true_answers_list[c] if score_matrix[r, c] > 0 else None
        )
    if calc_oracle_score:
        oracle_answers = sorted(
            list(true_answers.keys()), key=lambda z: true_answers[z], reverse=True
        )
        if isinstance(oracle_answers[0], frozenset):
            oracle_answers = [ans for (ans, *_) in oracle_answers]
        oracle_score, *_ = general_eval(
            pred_answers=oracle_answers,
            true_answers=true_answers,
            max_pred_answers=max_pred_answers,
            max_incorrect=max_incorrect,
            string_preprocessing=string_preprocessing,
            question_string=question_string,
            score_func=score_func,
            cluster_score_func=cluster_score_func,
            cluster_reduction_func=cluster_reduction_func,
            score_matrix_transformation=score_matrix_transformation,
            assign_cluster_scores=assign_cluster_scores,
            calc_oracle_score=False,
        )
        score /= oracle_score
    return EvalResult(
        score=score, score_matrix=score_matrix, answer_assignment=answer_assignment
    )


fast_money = partial(general_eval, max_pred_answers=1)

family_feud = partial(general_eval, max_incorrect=3)

family_feud_2_incorrect = partial(general_eval, max_incorrect=2)

family_feud_5_incorrect = partial(general_eval, max_incorrect=5)

set_intersection = partial(general_eval, assign_cluster_scores=False)

hard_set_intersection = partial(set_intersection, score_matrix_transformation=np.round)


max_answers = {
    f"Max Answers - {k}": partial(general_eval, max_pred_answers=k)
    for k in [1, 3, 5, 10]
}
max_incorrect = {
    f"Max Incorrect - {k}": partial(general_eval, max_incorrect=k) for k in [1, 3, 5]
}
exact_match_all_eval_funcs = {**max_answers, **max_incorrect}

# WordNet Similarity
wordnet_all_eval_funcs = {
    k: partial(v, score_func=wordnet_score, score_matrix_transformation=np.round)
    for k, v in exact_match_all_eval_funcs.items()
}

all_eval_funcs = {
    "exact_match": exact_match_all_eval_funcs,
    "wordnet": wordnet_all_eval_funcs,
}


# Direct implementations of some of the simpler algorithms,
# without the functional structure of the general setting.
# Useful for testing, in case something in the more general setting goes wrong.
def naive_family_feud(
    pred_answers: List[str],
    true_answers: Dict[str, int],
    *args,
    max_incorrect: int = 3,
    **kwargs
) -> float:
    pred_answers = pred_answers.copy()
    true_answers = true_answers.copy()
    score = 0
    max_score = sum(true_answers.values())
    incorrect = 0
    for i, answer in enumerate(pred_answers):
        try:
            score += true_answers.pop(answer)
        except KeyError:
            incorrect += 1
            if incorrect >= max_incorrect:
                break
    score /= max_score
    return score


def naive_fast_money(pred_answers, true_answers):
    pred_answers = pred_answers.copy()
    true_answers = true_answers.copy()
    score = true_answers.get(pred_answers[0], 0)
    score /= max(true_answers.values())
    return score
