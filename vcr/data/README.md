# VCR v1.0 readme
Packaged Dec 3, 2018. Contact `rowanz@cs.washington.edu` for help.


# Quick overview
The dataset consists of image/metadata files, and annotations. Each annotation file (`train.jsonl`, `val.jsonl`, and `test.jsonl`)is a jsonl file, where each line is a JSON object. The test file has the labels removed, because it'll be used in a public leaderboard. Stay tuned for that release.-

Here are the important things from the annotations:
* `objects`: a list of objects detected, for instance, ["person", "person", "horse", "horse", "horse", "horse"]
* `img_fn`: the filename of the image, within the `vcr1images` directory. 
* `metadata_fn`: the json metadata file for the image, within the `vcr1images` directory. 
* `question`: Tokenized version of the question. Detection tags are represented as lists, so one example might be
`["What", "are", [0,1], "doing", "?"]` which corresponds to `What are 0 and 1 doing?`, where 0 and 1 are indexes (starting at 0) into `objects`.
* `answer_choices`: A list of four answer choices, with the same format as `question`.
* `answer_label`: Which answer (0 to 3) is right in `answer_choices`.
* `rationale_choices`: A list of four rationale choices, with the same format as `question`.
* `rationale_label`: Which rationale (0 to 3) is right in `rationale_choices`.

Here's what you get from the JSON file located at `metadata_fn`:

* `boxes`: for each object, a `[x1, y1, x2, y2, score]` bounding box using the Detectron format. The score is the probability output by Detectron.
* `segms`: for each object, a list of polygons. Each polygon is a list of `[x,y]` points given by cv2.findContours.
* `width` and `height`: the dimensions of the image.

There are pretty much all that you need to get started. I'll try to upload code and a dataloader soon.

# Structure of the dataset
```
vcr1/
|-- vcr1images/
|   |-- VERSION.txt
|   |-- movie name, like movieclips_A_Fistful_of_Dollars
|   |   |-- image files, like Sv_GcxkmW4Y@29.jpg
|   |   |-- metadata files, like Sv_GcxkmW4Y@29.json
|-- train.jsonl
|-- val.jsonl
|-- test.jsonl
|-- README.md
```

## More detailed information for people who are really curious:

I put more information in the annotations in case they help. Some of these were taken out for the test set, to mask the labels.

The `VERSION.txt` contains the version info of the release.

* `img-id` A shorter ID of this image. It will look something like `{SPLIT}-{NUMBER}`
* `question_number`: The first question/answer/rationale for this `img-id`/`img-fn` has `qid=0`, the second has `qid=1`, etc.
* `annot_id`: The index of this question, answer, and rationale. It will look something like `{SPLIT}-{NUMBER}`
* `match_fold`: The fold that was used for adversarial matching. This is particularly relevant for the training set, as it was comprised of several adversarial matching folds that were joined together. It looks something like `{SPLIT}-{NUMBER}`
* `match_index`: The index of this question, answer, and rationale, within the `match-fold`. 
* `movie`: the movie that the image comes from. For MovieClips, we had to guess; for LSMDC, it's given by the dataset.
* `interesting_scores`: a list of interestingness scores given by the turkers, with one number per turker. 1 means exceptionally interesting, 0 means okay, and -1 means boring.
* `{question/answer/rationale}_orig`: For training and val, we also have the original ground-truth questions, answers, and rationales that the turkers provided. We processed them lightly (applying spell check and WordPiece tokenizing) to get the tokenized questions/correct answers/correct rationales that are in `question`, `answer_choices`, and `rationale_choices`
* `answer_likelihood`: How likely the turker said that their answer was - choices are (`unlikely`, `possible`, or `likely`).
* `{answer/rationale}_match_iter`: the Adversarial Matching iteration on which each answer was introduced. 0 means that the answer is ground truth, 1 means that it was the first wrong answer Adversarial Matching chose, 2 second, etc. For instance, `[2, 0, 3, 1]` means that the first answer was introduced on iteration 2, the second is the ground truth answer (`answer_label = 1`), and so on.
* `{answer/rationale}_sources`: the `match_index`es where we obtained the wrong answers from. To obtain a list of answers for `match_fold=2` for instance, we can just do:
    `answers_2 = {x['match_index']: x for x in dataset if x['match_fold'] == 2}`
    So now for any x in that list, `x['answer_sources'][i]` means that the `i`th answer came was originally the ground truth answer `answers_2[i]`. This also means that for all `j`, we have `answers_2[j]['answer_sources'][answers_2[j]['answer_label']] = j`.  
 
 
