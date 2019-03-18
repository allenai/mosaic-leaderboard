import argparse
import json
from typing import List
import random


def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def main(args):
    input_file = args.input_file
    output_file = args.output_file

    test_records = read_jsonl_lines(input_file)
    pred_answers = [random.choice(["1", "2"]) for r in  test_records]

    with open(output_file, "w") as f:
        for p in pred_answers:
            f.write(p)
            f.write("\n")
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A random baseline.')
    # Required Parameters
    parser.add_argument('--input_file', type=str, help='Location of input file', default=None)
    parser.add_argument('--output_file', type=str, help='Location of predictions', default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)
