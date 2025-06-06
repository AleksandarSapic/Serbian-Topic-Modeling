from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files="concatenated_dataset.jsonl", vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("model", "serbian")