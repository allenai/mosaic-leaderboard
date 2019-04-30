import argparse
import json

import numpy as np
import pandas as pd


def read_labels(labels_file: str) -> pd.DataFrame:
    labels = []
    with open(labels_file, 'r') as f:
        for l in f:
            item = json.loads(l)
            labels.append((item['answer_label'], item['rationale_label']))
    labels = pd.DataFrame(labels, index=[f'test-{i}' for i in range(len(labels))],
                          columns=['answer', 'rationale'])
    return labels


def score_submission(pred_labels, test_labels):
    assert pred_labels.shape[0] == test_labels.shape[0]

    # score answers
    answers = pred_labels[['answer_0', 'answer_1', 'answer_2', 'answer_3']].values.argmax(1)

    qa_accuracy = np.mean((answers == test_labels['answer']))

    print("Answer accuracy is {:.3f}".format(qa_accuracy))

    # score rationales (explanations for the answers)
    rationales = pred_labels[
        [f'rationale_conditioned_on_a{i}_{j}' for i in range(4) for j in range(4)]].values.reshape(
        (-1, 4, 4))

    rationales_conditioned_on_gt = rationales[
        np.arange(rationales.shape[0]), test_labels['answer'].values].argmax(1)

    qa2r_accuracy = np.mean((rationales_conditioned_on_gt == test_labels['rationale']))

    print("Rationale accuracy is {:.3f}".format(qa2r_accuracy))

    # Compute whether both the answer and the rational are correct
    is_right = ((answers == test_labels['answer']) & (
                rationales_conditioned_on_gt == test_labels['rationale']))

    q2ar_accuracy = np.mean(is_right)
    print("Combined accuracy is {:.3f}".format(q2ar_accuracy))
    return qa_accuracy, qa2r_accuracy, q2ar_accuracy


def main(labels_file, preds_file, metrics_output_file):

    # Read Labels
    gold_answers = read_labels(labels_file)

    # Read Predictions
    pred_answers = pd.read_csv(preds_file)

    # Score predictions
    qa_accuracy, qa2r_accuracy, q2ar_accuracy = score_submission(pred_answers, gold_answers)

    # Write to file
    results = {
        'accuracy_qa': qa_accuracy,
        'accuracy_qa2r': qa2r_accuracy,
        'accuracy_q2ar': q2ar_accuracy
    }
    with open(metrics_output_file, "w") as f:
        f.write(json.dumps(results))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate VCR predictions')
    # Required Parameters
    parser.add_argument('--labels_file', type=str, help='Location of test labels', default=None)
    parser.add_argument('--preds_file', type=str, help='Location of predictions', default=None)
    parser.add_argument('--metrics_output_file',
                        type=str,
                        help='Location of output metrics file',
                        default="metrics.json")

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args.labels_file, args.preds_file, args.metrics_output_file)
