# MT Tokenizer Bias
We analyze and mitigate gender bias in MT tokenizers.

#Walk through 

1. Follow the instructions at https://github.com/neulab/awesome-align to install awesome_align. 

2. ...

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
python src/update_vocabulary.py --src_lang <SRCL> --tgt_lang <TGTL> --translator <TRANS> --variants_dir <VARDIR> --average_embeddings --tokenizer_dir <TOKDIR> --model_dir <MODDIR>
```

where:

`translator` is the name of the original translation model from huggingface, by default `opus-nmt`.

`variants_dir` is the directory where the human translated professions are stored. Specifically, it should contain a file `<TGTL>_variants.json`.

`avarage_embeddings` is a flag that indicates whether the embeddings for the added words should be averaged from the existing emebddings in the model's vocabulary.

`tokenizer_dir` and `model_dir` are the directories where the tokenizer and the model are stored, respectively.

### Fine-tuning