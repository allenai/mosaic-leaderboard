import argparse
import json
from typing import List
import random
import os
import numpy as np


# Parse the input file from JSONL to a list of dictionaries.
def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def rand_prob_vector(size=4):
    v = np.random.uniform(0, 100, size=4)
    v = v / np.sum(v)
    return v


def main(input_dir, output_file):
    # Read the records from the test set.
    qa_test_records = read_jsonl_lines(os.path.join(input_dir, 'qa.jsonl'))
    qar_test_records = read_jsonl_lines(os.path.join(input_dir, 'qar.jsonl'))

    # Make predictions for each example in the test set.

    rows = []
    for qa, qar in zip(qa_test_records, qar_test_records):
        row = [qa['annot_id']]

        answer_probs = rand_prob_vector(len(qa['answer_choices']))
        row.extend([str(v) for v in answer_probs])

        for _ in answer_probs:
            row.extend([str(v) for v in rand_prob_vector(len(qar['rationale_choices']))])
        rows.append(row)

    # Write the predictions to the output file.
    fields = [
        "annot_id", "answer_0", "answer_1", "answer_2", "answer_3", "rationale_conditioned_on_a0_0",
        "rationale_conditioned_on_a0_1", "rationale_conditioned_on_a0_2",
        "rationale_conditioned_on_a0_3", "rationale_conditioned_on_a1_0",
        "rationale_conditioned_on_a1_1", "rationale_conditioned_on_a1_2",
        "rationale_conditioned_on_a1_3", "rationale_conditioned_on_a2_0",
        "rationale_conditioned_on_a2_1", "rationale_conditioned_on_a2_2",
        "rationale_conditioned_on_a2_3", "rationale_conditioned_on_a3_0",
        "rationale_conditioned_on_a3_1", "rationale_conditioned_on_a3_2",
        "rationale_conditioned_on_a3_3"]

    with open(output_file, "w") as f:
        f.write(",".join(fields))
        f.write("\n")
        for row in rows:
            f.write(",".join(row))
            f.write("\n")
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A random baseline.')
    parser.add_argument('--input-dir', type=str, required=True, help='Location of test data',
                        default=None)
    parser.add_argument('--output-file', type=str, required=True, help='Location of predictions',
                        default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args.input_dir, args.output_file)
