import re
import os
import nltk
from nltk.corpus import stopwords
from args import *

def remove_stopwords(dataset, language):
    print("cleaning dataset - removing stopwords")
    nltk.download("stopwords")
    stop_words = set(stopwords.words(language))

    removed_count = 0
    kept_count = 0
    cleaned_dataset = []
    for text in dataset:
        splitted_text = text.split()
        cleaned_text = []
        for word in splitted_text:
            if word.lower() not in stop_words:
                cleaned_text.append(word)
                kept_count += 1
            else:
                removed_count += 1
        temp_str = " ".join(cleaned_text).strip()
        cleaned_dataset.append(temp_str)

    print(f"stopwords removed: {removed_count}")
    print(f"stopwords kept: {kept_count}")
    return cleaned_dataset

def remove_rare_words(dataset, keep_frequency):
    print("cleaning dataset - removing rare words")
    word_count = {}
    for text in dataset:
        if type(text) == str:
            text = text.split()
        for word in text:
            if word.lower() not in word_count:
                word_count[word.lower()] = 1
            else:
                word_count[word.lower()] += 1

    removed_count = 0
    cleaned_dataset = []
    for text in dataset:
        cleaned_text = []
        if isinstance(text, str):
            text = text.split()
        for word in text:
            if word_count[word.lower()] >= keep_frequency:
                cleaned_text.append(word)
            else:
                removed_count += 1
        temp_str = " ".join(cleaned_text).strip()
        cleaned_dataset.append(temp_str)
    print(f"rare frequency words removed: {removed_count}")
    return cleaned_dataset

def clean_str(string):
    """
    Tokenization/string cleaning.
    Original implementation: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_and_tokenize_dataset(dataset):
    print("cleaning dataset - tokenizing")
    cleaned_dataset = []
    for text in dataset:
        cleaned_dataset.append(clean_str(text))
    return cleaned_dataset

def write_to_file(dataset, cleaned_dataset_dir, filename):
    os.makedirs(os.path.dirname(cleaned_dataset_dir), exist_ok=True)
    if filename is None:
        for i in range(len(dataset)):
            with open(cleaned_dataset_dir + str(i) + ".txt", "w") as f:
                f.write(cleaned_dataset[i])
    else:
        cleaned_dataset_str = "\n".join(cleaned_dataset)
        with open(cleaned_dataset_dir + filename, "w") as f:
            f.write(cleaned_dataset_str)
            print(f"dataset saved on {cleaned_dataset_dir + filename}")

def read_dataset(dataset_dir, dataset_name):
    dataset = []
    with open(dataset_dir + dataset_name + ".txt", "rb") as f:
        for line in f.readlines():
            dataset.append(line.decode('ISO-8859-1'))
    print(f"read a dataset with {len(dataset)} documents.")

    return dataset

if __name__ == "__main__":
    args = make_args_build_graph()
    print(args)

    accepted_datasets = ["20ng", "ohsumed", "R8", "R52"]
    if args.dataset not in accepted_datasets:
        raise Exception("dataset not valid.")

    dataset = read_dataset(args.corpus_dir, args.dataset)
    cleaned_dataset = clean_and_tokenize_dataset(dataset)
    cleaned_dataset = remove_rare_words(cleaned_dataset)
    cleaned_dataset = remove_stopwords(cleaned_dataset)

    write_to_file(cleaned_dataset, args.cleaned_dir, args.dataset + ".clean.txt")