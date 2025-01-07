"""
This module provides DataLoader functions for various NLP datasets.
The datasets included are:
1. IMDb (Text Classification - Easy)
2. AG News (Text Classification - Moderate)
3. Yelp Reviews (Text Classification - Difficult)
4. CoNLL-2003 (Named Entity Recognition - Standard)
5. SQuAD 2.0 (Question Answering - Standard)
"""

from torch.utils.data import DataLoader

from datasets import load_dataset


def get_imdb_dataloader(tokenizer, batch_size=32, split="train"):
    """
    Returns a DataLoader for the IMDb dataset (Text Classification).

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to preprocess the text.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        split (str, optional): Dataset split to load ('train' or 'test'). Defaults to 'train'.

    Returns:
        DataLoader: DataLoader for the IMDb dataset.
    """
    dataset = load_dataset("imdb", split=split)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=256
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_ag_news_dataloader(tokenizer, batch_size=32, split="train"):
    """
    Returns a DataLoader for the AG News dataset (Text Classification).

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to preprocess the text.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        split (str, optional): Dataset split to load ('train' or 'test'). Defaults to 'train'.

    Returns:
        DataLoader: DataLoader for the AG News dataset.
    """
    dataset = load_dataset("ag_news", split=split)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=256
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_yelp_reviews_dataloader(tokenizer, batch_size=32, split="train"):
    """
    Returns a DataLoader for the Yelp Reviews dataset (Text Classification).

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to preprocess the text.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        split (str, optional): Dataset split to load ('train' or 'test'). Defaults to 'train'.

    Returns:
        DataLoader: DataLoader for the Yelp Reviews dataset.
    """
    dataset = load_dataset("yelp_review_full", split=split)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=256
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_conll2003_dataloader(tokenizer, batch_size=32, split="train"):
    """
    Returns a DataLoader for the CoNLL-2003 dataset (NER).

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to preprocess the tokens.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        split (str, optional): Dataset split to load ('train', 'validation', or 'test').
            Defaults to 'train'.

    Returns:
        DataLoader: DataLoader for the CoNLL-2003 dataset.
    """
    dataset = load_dataset("conll2003", split=split)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            max_length=128,
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = dataset.map(tokenize_and_align_labels, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_squad_v2_dataloader(tokenizer, batch_size=32, split="train"):
    """
    Returns a DataLoader for the SQuAD 2.0 dataset (QA).

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to preprocess the text.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        split (str, optional): Dataset split to load ('train' or 'validation'). Defaults to 'train'.

    Returns:
        DataLoader: DataLoader for the SQuAD 2.0 dataset.
    """
    dataset = load_dataset("squad_v2", split=split)

    def preprocess_function(examples):  # pylint: disable=R0914
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )
        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            if len(answer["text"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)
                # Find the start and end token positions
                context_start = sequence_ids.index(1)
                context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
                # Adjust if the answer is out of the context
                if (
                    offset[context_start][0] > end_char
                    or offset[context_end][1] < start_char
                ):
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Find the start token
                    for idx, (start, end) in enumerate(offset):
                        if start <= start_char < end:
                            start_positions.append(idx)
                            break
                    else:
                        start_positions.append(0)
                    # Find the end token
                    for idx, (start, end) in enumerate(offset):
                        if start < end_char <= end:
                            end_positions.append(idx)
                            break
                    else:
                        end_positions.append(0)
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=dataset.column_names
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "start_positions", "end_positions"],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
