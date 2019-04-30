# VCR Random Baseline

This is an implementation of a random baseline for demonstration purposes.
If you are building your own solver this example will show you how to parse the input file and write predictions in the correct format.

## Running this example locally

To try out this random baseline, you must first download either the [training](https://storage.googleapis.com/ai2-alexandria/public/vcr/train.zip) or [validation](https://storage.googleapis.com/ai2-alexandria/public/vcr/val.zip) split of the VCR dataset.  Then you can run the random baseline with the following command.

```
python random_baseline.py --input-dir vcr/ --output-file predictions.csv
```
