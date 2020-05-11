# aNLI Random Baseline

This is an implementation of a random baseline for demonstration purposes.
If you are building your own solver this example will show you how to parse the input file and write predictions in the correct format.

## Running this example locally

To try out this random baseline, you must first download either the [train](https://storage.googleapis.com/ai2-alexandria/public/alpha-nli/train.jsonl) or [dev](https://storage.googleapis.com/ai2-alexandria/public/alpha-nli/valid.jsonl) split of the aNLI dataset.  Then you can run the random baseline with the following command.

```
python random_baseline.py --input-file train.jsonl --output-file predictions.lst
```

## Submitting to Leaderboard

1. Create a docker image
```
docker build -t alpha-nli-random-baseline .
```
2. Upload image to beaker
```
beaker image create --name alpha-nli-random-baseline alpha-nli-random-baseline
```
3. Navigate back to [Submission Creation Page](https://leaderboard.allenai.org/anli/submission/create).
`alpha-nli-random-baseline` should be visible in the `Beaker Image` suggestions. 
