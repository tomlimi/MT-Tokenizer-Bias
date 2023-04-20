# MT Tokenizer Bias
We analyze and mitigate gender bias in MT tokenizers.

#Walk through 

1. Follow the instructions at https://github.com/neulab/awesome-align to install awesome_align. 

2. ...

## Statistical Analysis

...

## Interventions

The following section describes how to extend the model's vocabulary to prevent from
splitting the profession words into multilple tokens. Next, we show how to fine-tune the model
on a gender balanced dataset to mitigate the bias.

### Updating Model's Vocabulary

In the update step the words from translated dataset are added to the model's vocabulary. Additionally,
the output embedding layer of the translation model is resized to include the representations
for the added words.

The two steps are performed by the following command:

```bash
python update_vocabulary.py --src_lang <SRCL> --tgt_lang <TGTL> --translator <TRANS> --variants_dir <VARDIR> --average_embeddings --tokenizer_dir <TOKDIR> --model_dir <MODDIR>
```

where:

`translator` is the name of the original translation model from huggingface, by default `opus-nmt`.

`variants_dir` is the directory where the human translated professions are stored. Specifically, it should contain a file `<TGTL>_variants.json`.

`avarage_embeddings` is a flag that indicates whether the embeddings for the added words should be averaged from the existing emebddings in the model's vocabulary.

`tokenizer_dir` and `model_dir` are the directories where the tokenizer and the model are stored, respectively.

### Fine-tuning

(For fine tuning we use our gender balance dataset, download it or follow the instruction below to create it)

The model is fine-tuned on the set of sentences containing simple sentences with profession names.
To fine tune the model with the default parameters run:

```bash
source ../scripts/fine_tune.sh <TGTL> <KEEP_PROFESSIONS> <TRANSLATOR>
```

where
<KEEP_PROFESSIONS> is the flag whether to use the model with the updated vocabulary (value: 1) or not (value: 0).

<TRANSLATOR> is the name of the original translation model from huggingface, by default `opus-mt` or `mbart50`.


=======

## Gender balanced dataset creation

To create the dataset run the command:

```bash
python prepare_balanced_dataset.py --src_lang <SRCL> --tgt_lang <TGTL> --variants_dir <VARDIR> --out_dir <OUTDIR>
```

where:

`variants_dir` is the directory where the human translated professions are stored. Specifically, it should contain a file `<TGTL>_variants.json`.

`out_dir` is the directory where the balanced dataset will be stored.

The data is saved in the new-line seperated list of jason strings (jsonl format).
