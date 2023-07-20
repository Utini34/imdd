import re
import string
import torch
import numpy as np
import pandas as pd

from itertools import chain
from datasets import load_dataset, DatasetDict, ClassLabel, Features, Value
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import T5Tokenizer


def cleaning_sentiment_data_custom_data(dataset):
    dataset = dataset.rename_columns({"review": "text", "sentiment": "label"})
    dataset = dataset.filter(lambda example: example["label"] is not None)
    dataset = dataset.map(map_label)
    return dataset


def map_label(example):
    # Define the label dictionary
    label_dict = {"positive": 1, "negative": 0}
    example["label"] = label_dict.get(example["label"], example["label"])
    return example


def load_data(name, split):
    if name == "sentiment_data_custom":
        try:
            # data schema
            class_label = ClassLabel(names=['neg', 'pos'])
            schema = Features(
                {
                    'text': Value('string'),  # Example feature
                    'label': class_label,  # Assign the ClassLabel to the 'label' feature
                }
            )
            dataset = load_dataset(path="./data/sentiment_data_custom", split=split, features=schema)
        except:
            dataset = load_dataset(path="./data/sentiment_data_custom", split=split)
            dataset = cleaning_sentiment_data_custom_data(dataset)
    else:
        dataset = load_dataset(name, split=split)
    return dataset


def flatten_list(nested_list):
    return list(chain.from_iterable(nested_list))


def split_dataset(dataset, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
    # Calculate the number of samples for each split
    num_samples = len(dataset)
    train_samples = int(train_ratio * num_samples)
    validation_samples = int(validation_ratio * num_samples)
    test_samples = int(test_ratio * num_samples)

    # Perform the splits
    train_and_test = dataset.train_test_split(
        train_size=(train_samples + validation_samples), test_size=test_samples
    )
    train_and_validation = train_and_test["train"].train_test_split(
        train_size=train_samples, test_size=validation_samples
    )
    train = train_and_validation["train"]
    validation = train_and_validation["test"]
    test = train_and_test["test"]

    # Combine the splits into a DatasetDict
    split_datasets = DatasetDict({"train": train, "val": validation, "test": test})

    return split_datasets


def preprocess_data(
    data,
    label_encoder,
    glove_embedding,
    tokenizer_t5,
    max_sequence_length=200,
    num_classes=2,
):
    def _preprocess(sample):
        text = clean_text(sample["text"])
        tokens = word_tokenize(text)
        tokens = pad_sequence(tokens, max_length=max_sequence_length)
        encoded_tokens = encode_tokens(tokens, label_encoder=label_encoder)
        input_ids, attention_mask, target_input_ids = encode_sample_t5(
            sample, tokenizer_t5, max_sequence_length=max_sequence_length
        )
        input_glove_vectors = encode_sample_glove(encoded_tokens, glove_embedding)
        # label_onehot = encode_label_onehot(sample["label"], num_classes=num_classes)
        return {
            "cleaned_text": text,
            "tokens": tokens,
            "encoded_tokens": encoded_tokens,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_input_ids,
            "input_glove_vectors": input_glove_vectors,
            # "label_onehot": label_onehot,
        }

    # Clean and tokenize text
    data = data.map(_preprocess)

    return data


def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word not in stop_words])

    return text


def pad_sequence(tokens, max_length):
    if len(tokens) >= max_length:
        return tokens[:max_length]
    else:
        return tokens + ["<PAD>"] * (max_length - len(tokens))


def create_dataloader(dataset, batch_size):
    loader = DataLoader(
        dataset.with_format("torch"), batch_size=batch_size, shuffle=True
    )
    return loader


def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line_parts = line.split()
            word = line_parts[0]
            vector = np.asarray(line_parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings


def add_oov_token(embeddings, strategy="random", embedding_dim=100):
    special_tokens = ["<PAD>", "<OOV>"]
    for special_token in special_tokens:
        if strategy == "mean":
            all_vectors = embeddings.values()
            oov_vector = np.mean(list(all_vectors), axis=0)
        else:
            oov_vector = np.random.uniform(-1, 1, embedding_dim)
        embeddings[special_token] = oov_vector


def build_label_encoder(vocabs):
    # Encode the labels
    label_encoder = LabelEncoder()
    all_tokens = list(set(vocabs + ["<PAD>", "<OOV>"]))
    label_encoder.fit(all_tokens)
    return label_encoder


def encode_tokens(tokens, label_encoder):
    tokens = [token if token in label_encoder.classes_ else "<OOV>" for token in tokens]
    return label_encoder.transform(tokens)


def glove_to_tensor(label_encoder, glove_embeddings, embedding_dim=50):
    # Initialize the tensor with zeros
    vocab_size = len(label_encoder.classes_) + 1
    embedding_tensor = torch.zeros(vocab_size, embedding_dim)

    # Iterate through the tokenizer's word_index
    for idx, word in enumerate(label_encoder.classes_):
        word_embedding = glove_embeddings.get(word)
        if word_embedding is not None:
            embedding_tensor[idx] = torch.tensor(word_embedding, dtype=torch.float32)

    return embedding_tensor


def binary_accuracy(preds, y, threshold=0.5):
    rounded_preds = (preds >= threshold).int()
    correct = rounded_preds == y
    acc = correct.sum() / len(correct)
    return acc


def encode_sample_t5(sample, tokenizer, max_sequence_length):
    input_text = "sentiment: " + sample["text"]
    target_text = "pos" if sample["label"] == 1 else "neg"
    input_encoding = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_sequence_length,
        truncation=True,
        padding='max_length',
    )
    target_encoding = tokenizer(
        target_text,
        return_tensors="pt",
        max_length=3,
        truncation=True,
        padding='max_length',
    )
    return (
        input_encoding["input_ids"].squeeze(),
        input_encoding["attention_mask"].squeeze(),
        target_encoding["input_ids"].squeeze(),
    )


def encode_sample_glove(encoded_tokens, glove_embedding):
    return glove_embedding[encoded_tokens].reshape(-1).numpy()  # [L*D]


def encode_label_onehot(class_label, num_classes):
    return np.eye(num_classes)[class_label]


def load_content(path):
    with open(path, "r") as f:
        content = f.read()
    return content
