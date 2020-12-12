import math
import json
import argparse
import statistics
from typing import List
from sklearn.metrics import accuracy_score, auc

from data_processing import load_data_from_jsonl, load_predictions_from_jsonl
from evaluation import multiple_evals, all_eval_funcs


def main(args):
    labels_file = args.labels_file
    preds_file = args.preds_file
    similarity_function="wordnet"
    metrics_output_file = args.metrics_output_file

    """Run all evaluation metrics on model outputs"""

    print(f"Using {similarity_function} similarity.", flush=True)
    targets = load_data_from_jsonl(labels_file)
    predictions = load_predictions_from_jsonl(preds_file)
    results = multiple_evals(
        eval_func_dict=all_eval_funcs[similarity_function],
        question_data=targets,
        answers_dict=predictions,
    )

    print(results)
    output_scores = {}
    for name, eval_details in results.items():
        print(name)
        eval_score = statistics.mean(x.score for x in eval_details.values())
        output_scores[name] = eval_score

    with open(metrics_output_file, "w") as f:
        f.write(json.dumps(output_scores))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate ProtoQA predictions')
    # Required Parameters
    parser.add_argument('--labels_file', type=str, help='Location of labels', default='/Users/ronanlb/ai2/mosaic/mosaic-leaderboard/mosaic-leaderboard/protoqa/evaluator/labels.jsonl')
    parser.add_argument('--preds_file', type=str, help='Location of predictions', default='/Users/ronanlb/ai2/mosaic/mosaic-leaderboard/mosaic-leaderboard/protoqa/evaluator/predictions.jsonl')
    parser.add_argument('--metrics_output_file',
                        type=str,
                        help='Location of output metrics file',
                        default="metrics.json")

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)
