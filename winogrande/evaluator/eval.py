import argparse
import json
from typing import List
from sklearn.metrics import accuracy_score


def read_lines(input_file: str) -> List[str]:
    lines = []
    with open(input_file, "rb") as f:
        for l in f:
            lines.append(l.decode().strip())
    return lines


def main(args):
    labels_file = args.labels_file
    preds_file = args.preds_file
    metrics_output_file = args.metrics_output_file

    gold_answers = read_lines(labels_file)
    pred_answers = read_lines(preds_file)

    if len(gold_answers) != len(pred_answers):
        raise Exception("The prediction file does not contain the same number of lines as the "
                        "number of test instances.")

    accuracy = accuracy_score(gold_answers, pred_answers)
    results = {
        'accuracy': accuracy
    }
    with open(metrics_output_file, "w") as f:
        f.write(json.dumps(results))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate WinoGrande predictions')
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
    main(args)
