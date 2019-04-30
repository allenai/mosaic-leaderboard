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


def score_submission(example_submission, test_labels):
    assert example_submission.shape[0] == test_labels.shape[0]

    # score answers
    answers = example_submission[['answer_0', 'answer_1', 'answer_2', 'answer_3']].values.argmax(1)

    qa = np.mean((answers == test_labels['answer']))

    print("Answer acc is {:.3f}".format(qa))

    # score rationales
    rationales = example_submission[
        [f'rationale_conditioned_on_a{i}_{j}' for i in range(4) for j in range(4)]].values.reshape(
        (-1, 4, 4))

    rationales_conditioned_on_gt = rationales[
        np.arange(rationales.shape[0]), test_labels['answer'].values].argmax(1)

    qa2r = np.mean((rationales_conditioned_on_gt == test_labels['rationale']))

    print("Rationale acc is {:.3f}".format(qa2r))

    # Combine
    is_right = ((answers == test_labels['answer']) & (
                rationales_conditioned_on_gt == test_labels['rationale']))

    q2ar = np.mean(is_right)
    print("Combined acc is {:.3f}".format(q2ar))
    return qa, qa2r, q2ar


def main(labels_file, preds_file, metrics_output_file):

    # Read Labels
    gold_answers = read_labels(labels_file)

    # Read Predictions
    pred_answers = pd.read_csv(preds_file)

    # Score predictions
    qa, qa2r, q2ar = score_submission(pred_answers, gold_answers)

    # Write to file
    results = {
        'accuracy_qa': qa,
        'accuracy_qa2r': qa2r,
        'accuracy_q2ar': q2ar
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
