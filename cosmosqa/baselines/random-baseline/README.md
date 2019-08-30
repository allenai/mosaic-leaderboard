# CosmosQA Random Baseline

This is an implementation of a random baseline for demonstration purposes.
If you are building your own solver this example will show you how to parse the input file and write predictions in the correct format.

## Running this example locally

To try out this random baseline, you must first download the [train and valid](https://storage.googleapis.com/ai2-mosaic/public/cosmosqa/cosmosqa-train-valid.zip) data. Then you can run the random baseline with the following command.

```
python random_baseline.py --input-file train.jsonl --output-file predictions.lst
```
