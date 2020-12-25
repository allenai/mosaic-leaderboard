import warnings
from difflib import SequenceMatcher
from functools import partial
from itertools import product
from typing import *

import nltk
import numpy as np
from more_itertools import partitions
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from scipy.optimize import linear_sum_assignment

__all__ = [
    "all_pairs_scores",
    "get_optimal_score",
    "exact_match",
    "longest_common_substring_score",
    "longest_common_subsequence_score",
    "wordnet_synsets_score",
    "wordnet_partition_score",
    "wordnet_score",
    "wup_similarity_wrapper",
    "wordnet_wup_synset_score",
    "wordnet_wup_partition_score",
    "wordnet_wup_score",
    "cluster_score",
    "scale_score_matrix_by_cluster_scores",
    "limit_total_wrong",
]

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

EN_STOPWORDS = frozenset(stopwords.words("english"))


def all_pairs_scores(
    a: Union[str, Iterable],
    b: Union[str, Iterable],
    score_func: Callable,
    reduction_func: Callable = lambda z: z,
    preprocess_func: Callable = lambda z: [z] if isinstance(z, str) else z,
) -> Union[np.ndarray, float]:
    """
    Generic function for pairwise comparisons. Takes strings or iterables a and b and
    a score function and returns the matrix of all pairwise scores between a and b.

    :param a: Typically a string or iterable of strings to compare with b.
    :param b: Typically a string or iterable of strings to compare with a.
    :param score_func: Function which accepts two arguments (a,b) and returns their score in [0,1]
    :param reduction_func: Function which accepts a matrix and (typically) returns a scalar
    :param preprocess_func: Function which is run on both a and b prior to anything else
    :param kwargs: passed on to the score_func
    :return: Matrix of pairwise scores or output of reduction function on this matrix
    """
    a, b = preprocess_func(a), preprocess_func(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    score_matrix = np.zeros((len(a), len(b)))
    for (a_idx, a_val), (b_idx, b_val) in product(enumerate(a), enumerate(b)):
        score_val = score_func(a_val, b_val)
        score_matrix[a_idx, b_idx] = score_val
        if not (0 <= score_val <= 1):
            warnings.warn(
                f"Score function did not return a value in [0,1]: "
                f"score_func({a_val}, {b_val}) = {score_val} with type {type(score_val)}"
            )
    return reduction_func(score_matrix)


##########################################################################
# Functions which take in a score matrix and return the actual score
##########################################################################
def get_optimal_score(score_matrix: np.ndarray) -> Tuple[float, List[int], List[int]]:
    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return score_matrix[row_ind, col_ind].sum(), row_ind, col_ind


#################################################################
# Functions which take a single pred_answer and true_answer,
# and return a score.
##################################################################
def exact_match(pred_answer: str, true_answer: str) -> float:
    return float(pred_answer == true_answer)


def longest_common_substring_score(pred_answer: str, true_answer: str) -> float:
    sm = SequenceMatcher(None, pred_answer, true_answer)
    match = sm.find_longest_match(0, len(pred_answer), 0, len(true_answer))
    return match.size / max(len(pred_answer), len(true_answer))


def longest_common_subsequence_score(pred_answer: str, true_answer: str) -> float:
    sm = SequenceMatcher(None, pred_answer, true_answer)
    lcsubseq_size = sum([block.size for block in sm.get_matching_blocks()])
    return lcsubseq_size / max(len(pred_answer), len(true_answer))


wordnet_synsets_score = partial(
    all_pairs_scores,
    score_func=lambda a, b: 1.0 if a == b else 0.0,
    reduction_func=np.max,
    preprocess_func=lambda z: wn.synsets(z.replace(" ", "_")),
)
wordnet_synsets_score.__name__ = "wordnet_synsets_score"
wordnet_synsets_score.__docs__ = (
    "Takes in a pair of strings which each get mapped to corresponding synsets, "
    "then returns the max similarity score between over any pairing of these synsets."
)


##########################################################################
# Functions which take an iterable of pred_answers and true_answers,
# calculate a score matrix of the pairwise combinations, and return a float
##########################################################################
wordnet_partition_score = partial(
    all_pairs_scores,
    score_func=lambda a, b: max(wordnet_synsets_score(a, b), exact_match(a, b)),
    reduction_func=lambda z: get_optimal_score(z)[0] / max(z.shape),
)
wordnet_partition_score.__name__ = "wordnet_partition_score"
wordnet_partition_score.__docs__ = (
    "Takes in a pair of partitions (List[str]) and computes the optimal matching between "
    "the parts of these partitions based on WordNet synsets or exact string match."
)


def wordnet_score(
    pred_answer: str,
    true_answer: str,
    score_func: Callable = wordnet_partition_score,
    reduction_func: Callable = np.max,
    *,
    stopwords=EN_STOPWORDS
):
    """
    WordNet score function for a predicted answer string and a true answer string.
    Takes in strings, tokenizes them, and returns the score which corresponds to the optimal
    partition of the original strings.
    """

    def _preprocess(z, stopwords=stopwords):
        tokens = [tok for tok in word_tokenize(z) if tok not in stopwords]
        parts = [[" ".join(tokens) for tokens in part] for part in partitions(tokens)]
        return parts

    return all_pairs_scores(
        pred_answer, true_answer, score_func, reduction_func, _preprocess
    )


# Wu-Palmer Similarity (https://linguistics.stackexchange.com/questions/9084/what-do-wordnetsimilarity-scores-mean)
def wup_similarity_wrapper(*args, **kwargs):
    sim = wn.wup_similarity(*args, **kwargs)
    if sim is None:
        sim = 0.0
    return sim


wordnet_wup_synset_score = partial(
    wordnet_synsets_score, score_func=wup_similarity_wrapper
)
wordnet_wup_partition_score = partial(
    wordnet_partition_score,
    score_func=lambda a, b: max(wordnet_wup_synset_score(a, b), exact_match(a, b)),
)
wordnet_wup_score = partial(wordnet_score, score_func=wordnet_wup_partition_score)
wordnet_wup_score.__name__ = "wordnet_wup_score"


def cluster_score(
    pred_answers: List[str],
    true_answers: Union[Dict[str, int], Dict[frozenset, int]],
    question_string: str,
    score_func: Callable = exact_match,
    cluster_reduction_func: Callable = np.max,
) -> np.ndarray:
    true_ans, *_ = true_answers
    if isinstance(true_ans, frozenset):
        score_func = partial(
            all_pairs_scores,
            score_func=score_func,
            reduction_func=cluster_reduction_func,
        )
    return all_pairs_scores(pred_answers, true_answers, score_func)


##########################################################################
# Functions which take in a score matrix and return an augmented
# score matrix
##########################################################################
def scale_score_matrix_by_cluster_scores(
    score_matrix: np.ndarray, cluster_scores: List[int]
) -> np.ndarray:
    return score_matrix * np.array(cluster_scores)[None]


def limit_total_wrong(score_matrix: np.ndarray, k: int) -> np.ndarray:
    answer_scores = score_matrix.max(axis=1)
    incorrect = 0
    for i, a in enumerate(answer_scores):
        if a == 0:
            incorrect += 1
            if incorrect >= k:
                return score_matrix[: i + 1]
    return score_matrix
