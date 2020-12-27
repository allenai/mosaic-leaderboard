from sklearn.gaussian_process import GaussianProcessRegressor
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import BertModel, BertTokenizer
import torch

import time
import numpy
from warnings import warn
from typing import *


def transform_question(question):
    question = question.lower()
    question = question.replace(".", "")
    question = question.replace(":", "")
    question = question.replace("?", "")
    question = question.replace("someone", "one person")
    question = question.replace("someplace", "one place")
    if "name something" in question:
        question = question.replace("name something", "one thing")
        question += " is"
    elif "tell me something" in question:
        question = question.replace("tell me something", "one thing")
        question += " is"
    elif "name a " in question:
        question = question.replace("name a ", "one ")
        question += " is"
    elif "name an " in question:
        question = question.replace("name an ", "one ")
        question += " is"
    elif "name" in question:
        question = question.replace("name", "")
        question += " is"
    elif question.startswith("tell me a "):
        question = question.replace("tell me a ", "one ")
        question += " is"
    elif question.startswith("tell me an "):
        question = question.replace("tell me an ", "one ")
        question += " is"
    elif question.startswith("what "):
        question = question.replace("what", "one")
        question += " is"
    elif question.startswith("give me a "):
        question = question.replace("give me a ", "one ")
        question += " is"
    elif question.startswith("tell me "):
        question = question.replace("tell me ", "")
        question += " is"
    elif "which" in question:
        question = question.replace("which", "one")
        question += " is"
    elif "what" in question:
        question = question.replace("what", "one")
        question += " is"
    elif "how can you tell" in question:
        question = question.replace("how can you tell", "one way to tell")
        question += " is"
    else:
        question = "Q: " + question + "? A: "
    return question


class MLMSim(torch.nn.Module):
    def __init__(self, threshold=0.1, modelname="roberta-large"):
        super(MLMSim, self).__init__()
        if modelname.startswith("roberta-"):
            self.model = RobertaModel.from_pretrained(modelname)
            self.TOKENIZER = RobertaTokenizer.from_pretrained(modelname)
        elif modelname.startswith("bert-base-uncased"):
            self.model = BertModel.from_pretrained(modelname)
            self.TOKENIZER = BertTokenizer.from_pretrained(modelname)

        self.model.requires_grad = False
        self.pad_token = self.TOKENIZER.pad_token_id
        self.threshold = threshold

        self.mapping_log = open("log.txt", "w")

    def get_vector_list(self, question, all_answers):
        with torch.no_grad():
            question = transform_question(question)
            qlength = len(self.TOKENIZER.encode(question)) - 1
            id_lists = [self.TOKENIZER.encode(question + " " + a) for a in all_answers]
            max_for_padding = max([len(x) for x in id_lists])
            padded_ids = []
            for qa_pair in id_lists:
                while len(qa_pair) < max_for_padding:
                    qa_pair.append(self.pad_token)
                padded_ids.append(qa_pair)
            all_examples_in_cluster = (
                self.model(torch.LongTensor(padded_ids))[0].cpu().data.numpy()
            )
            al = all_examples_in_cluster[:, qlength:, :]
            return numpy.mean(al, 1)

    def train_models(self, question_string, all_predicted_answers, true_answers):
        all_answers = []
        cluster_lists = {}
        answer_list = []
        seen_words = dict()
        for cid, each_cluster in enumerate(true_answers):
            all_answers.extend(each_cluster)
            cluster_lists[cid] = each_cluster
            answer_list.extend([cid] * len(each_cluster))
            for word in each_cluster:
                seen_words[word] = cid
        starttime = time.time()
        vectorized_answers = self.get_vector_list(
            question_string, all_answers + all_predicted_answers
        )
        training_vectors = vectorized_answers[: len(all_answers), :]
        test_vectors = vectorized_answers[len(all_answers) :, :]
        with torch.no_grad():
            ### train a dictionary of predictors for each clusters
            predictors = {}
            number_of_test_examples = int(test_vectors.shape[0])
            for each_group in list(set(answer_list)):
                clf = GaussianProcessRegressor(alpha=1e-7, n_restarts_optimizer=4)
                y = numpy.array(
                    [float(int(bool(x == each_group))) for x in answer_list]
                )
                clf.fit(training_vectors, y)
                predictors[each_group] = clf

            ### Get predictions for each cluster, and allocate them to each test example
            predictions_dictionary = {}
            for each_group in sorted(predictors):
                p = predictors[each_group].predict(test_vectors)
                for each_example in range(number_of_test_examples):
                    predictions_dictionary[each_example] = predictions_dictionary.get(
                        each_example, []
                    ) + [(p[each_example], each_group)]

            scoring_map = numpy.zeros((number_of_test_examples, max(answer_list) + 1))
            for example in predictions_dictionary:
                ### hard-code mapping to a cluster if we've already seen that word
                if all_predicted_answers[example] in seen_words:
                    scoring_map[example, seen_words[all_predicted_answers[example]]] = 1
                    selected = seen_words[all_predicted_answers[example]]
                else:
                    ### Otherwise, for each test example, if the highest-scoring example exceeds self.threshold, assign to that cluster
                    best = sorted(predictions_dictionary[example])[-1]
                    if best[0] > self.threshold:
                        scoring_map[example, best[1]] = 1.0
                        selected = best[1]
                    else:
                        selected = -1
                self.mapping_log.write(
                    "\t".join(
                        [
                            question_string,
                            all_predicted_answers[example],
                            str(selected),
                            "_".join(cluster_lists.get(selected, [])),
                        ]
                    )
                    + "\n"
                )
            return scoring_map


class ClusterScoreConsideringWholeCluster:
    """
    For MLM similarity, we do not break up the word into tokens, and the different answer clusters must all be
    considered simultaneously, and thus this single class takes the place of the entire cluster scoring function.
    """

    def __init__(self):
        self.mlm_similarity_scorer = None

    def __call__(
        self,
        pred_answers: List[str],
        true_answers: Union[Dict[str, int], Dict[frozenset, int]],
        question_string: str,
        score_func: Optional[Callable] = None,
        cluster_reduction_func: Optional[Callable] = None,
    ) -> np.ndarray:
        # Note: The function signature here has been expanded to contain the same parameters as the cluster scoring
        # function, however many of these parameters are ignored. In the event that they have actually been passed,
        # we throw a warning.
        if score_func is not None:
            warn(
                "MLM similarity was incorrectly called with an explicit score_func, this is being ignored."
            )
        if cluster_reduction_func is not None:
            warn(
                "MLM similarity was incorrectly called with an explicit cluster_reduction_func, this is being ignored."
            )

        if self.mlm_similarity_scorer is None:
            self.mlm_similarity_scorer = MLMSim()
        postulated_output = self.mlm_similarity_scorer.train_models(
            question_string, pred_answers, true_answers
        )
        return postulated_output
