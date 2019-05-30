import argparse
import json
from typing import List
import random

# Parse the input file from JSONL to a list of dictionaries.
def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def main(input_file, output_file):
    # Read the records from the test set.
    test_records = read_jsonl_lines(input_file)

    # Make predictions for each example in the test set.
    predicted_answers = [random.choice(["0", "1", "2", "3"]) for r in  test_records]

    # Write the predictions to the output file.
    with open(output_file, "w") as f:
        for p in predicted_answers:
            f.write(p)
            f.write("\n")
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A random baseline.')
    parser.add_argument('--input-file', type=str, required=True, help='Location of test records', default=None)
    parser.add_argument('--output-file', type=str, required=True, help='Location of predictions', default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args.input_file, args.output_file)
