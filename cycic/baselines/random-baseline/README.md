# CycIC Random Baseline

This is an implementation of a random baseline for demonstration purposes.
If you are building your own solver this example will show you how to parse the input file and write predictions in the correct format.

## Running this example locally

To try out this random baseline, you must first download either the [train and dev](https://storage.googleapis.com/ai2-mosaic/public/cycic/CycIC-train-dev.zip) split of the CycIC dataset.  Then you can run the random baseline with the following command.

```
python random_baseline.py --input-file train.jsonl --output-file predictions.lst
```

## Submitting to Leaderboard

1. Create a docker image
```
docker build -t cycic-random-baseline .
```
2. Upload image to beaker
```
beaker image create --name cycic-random-baseline cycic-random-baseline
```
3. Navigate back to [Submission Creation Page](https://leaderboard.allenai.org/cycic/submission/create).
`cycic-random-baseline` should be visible in the `Beaker Image` suggestions. 
