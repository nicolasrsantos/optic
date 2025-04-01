# %% [markdown]
# ### Notebook based on the tutorial available on HuggingFace
# https://huggingface.co/docs/transformers/en/tasks/sequence_classification

# %%
from utils import *
import os
from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
import numpy as np
import evaluate
import time

VALID_DATASETS = ["ohsumed", "R8", "R52", "mr", "SST1", "SST2", "TREC", "WebKB"]

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # GPU node id
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disable parallel gpu computing

# %%
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %%
seed = 42
set_seed(seed)

# %%
def load_configs(dataset):
    if dataset == "R8":
        output_dim = 8
    elif dataset == "R52":
        output_dim = 52
    elif dataset == "ohsumed":
        output_dim = 23
    elif dataset == "mr":
        output_dim = 2
    elif dataset == "TREC":
        output_dim = 6
    elif dataset == "WebKB":
        output_dim = 8
    elif dataset == "SST1":
        output_dim = 5
    elif dataset == "SST2":
        output_dim = 2
    elif dataset == "20ng":
        output_dim = 20

    return output_dim

# %%
data_dir = "data/"
dataset_name = "TREC"
out_dim = load_configs(dataset_name)


# %%
def get_data(dataset_name):
    cleaned_dir = "data/cleaned/"
    train_test_info_dir = "data/train_test_info/"
    if dataset_name not in VALID_DATASETS:
        raise Exception(
            "dataset not valid.\nsupported datasets {accepted_datasets}"
        )

    # Read dataset and embedding files
    cleaned_dataset = dataset_name + ".clean"
    dataset = read_file(cleaned_dir, cleaned_dataset)
    train_test_info = read_file(train_test_info_dir, dataset_name)

    return dataset, train_test_info

# %%
cur_graph_dir = os.path.join(data_dir, dataset_name)
dataset, train_test_info = get_data(dataset_name)

# %%
# check if splits are cached
cur_graph_dir = f"{data_dir}graphs/{dataset_name}/{dataset_name}"
train_ids_filename = cur_graph_dir + ".train_ids"
test_ids_filename = cur_graph_dir + ".test_ids"
cached_ids_available = exists(train_ids_filename) and exists(test_ids_filename)

# Get training and test information
doc_name_list = []
doc_train_list = []
doc_test_list = []
for tti in train_test_info:
    doc_name_list.append(tti.strip())

    if not cached_ids_available:
        # if splits are not cached -> gotta build from scratch
        temp = tti.split()
        if temp[1].find("train") != -1:
            doc_train_list.append(tti.strip())
        if temp[1].find("test") != -1:
            doc_test_list.append(tti.strip())

if cached_ids_available:
    # If cached files are available -> load them
    # to avoid different splits for the same dataset
    print("loading cached id files to build graph...")
    with open(train_ids_filename, "rb") as f:
        train_ids = pkl.load(f)
    with open(test_ids_filename, "rb") as f:
        test_ids = pkl.load(f)
else:
    print("missing id file(s). building from scratch...")
    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    random.shuffle(train_ids)

    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    random.shuffle(test_ids)

    # caching ids
    with open(train_ids_filename, "wb") as f:
        pkl.dump(train_ids, f)
    with open(test_ids_filename, "wb") as f:
        pkl.dump(test_ids, f)

# shuffle dataset
ids = train_ids + test_ids
shuffled_doc_name_list = []
shuffled_dataset = []
for id in ids:
    shuffled_doc_name_list.append(doc_name_list[int(id)])
    shuffled_dataset.append(dataset[int(id)])


# Get labels
y_unmapped = []
for doc_meta in shuffled_doc_name_list:
    temp = doc_meta.split("\t")
    y_unmapped.append(temp[2])
label2id = {label:i for i, label in enumerate(set(y_unmapped))}
id2label = {i:label for i, label in enumerate(set(y_unmapped))}

y = []
for label in y_unmapped:
    y.append(label2id[label])


train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size

ids = train_ids + test_ids
masks_train = ids[0:real_train_size]
masks_val = ids[real_train_size:real_train_size+val_size]
masks_test = ids[train_size:]

assert len(shuffled_dataset) == len(y)

shuffled_dataset = np.array(shuffled_dataset)
y = np.array(y)

# %%
train_data = {
    "text": [doc for doc in shuffled_dataset[masks_train]],
    "label": [label for label in y[masks_train]]
}
train_data = Dataset.from_dict(train_data)
val_data = {
    "text": [doc for doc in shuffled_dataset[masks_val]],
    "label": [label for label in y[masks_val]]
}
val_data = Dataset.from_dict(val_data)
test_data = {
    "text": [doc for doc in shuffled_dataset[masks_test]],
    "label": [label for label in y[masks_test]]
}
test_data = Dataset.from_dict(test_data)

# %%
dataset_dict = DatasetDict({
    'train': train_data,
    'val': val_data,
    'test': test_data
})

# %%
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=out_dim, id2label=id2label, label2id=label2id
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

# %%
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# %%
tokenized_data = dataset_dict.map(preprocess_function, batched=True)

# %%
training_args = TrainingArguments(
    output_dir=f"models/{dataset_name}",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    seed=seed,
    dataloader_num_workers=4
)

# %%
def experiment(data, training_args, model, data_collator, compute_metrics):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    _, _, metrics = trainer.predict(data["test"])
    
    return metrics["test_accuracy"]

# %%
n_runs = 10
accs = []

start = time.time_ns()
for run in range(n_runs):
    acc = experiment(tokenized_data, training_args, model, data_collator, compute_metrics)
    print(f"Run: {run + 1}, Test Acc: {acc}")
    accs.append(acc)

std = np.std(accs)
avg = np.mean(accs)
print(f"{avg*100:.2f}$\pm${std*100:.2f}")
print(f"Elapsed time: {(time.time_ns()-start)/1e9}")

# %%
dataset_name = "bla"
avg = 10
std = 20

# %%
with open("results_bert.csv", "a") as f:
    f.write(f"{dataset_name},{avg},{std}\n")