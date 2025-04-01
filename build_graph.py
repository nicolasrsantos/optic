import sys
import time
import random
import networkx as nx
import os
from networkx.algorithms import bipartite
from sentence_transformers import SentenceTransformer

from utils import *
from args import *

VALID_DATASETS = ["20ng", "ohsumed", "R8", "R52", "mr", "SST1", "SST2", "TREC", "WebKB"]


def get_data(args):
    if args.dataset not in VALID_DATASETS:
        raise Exception(
            "dataset not valid.\nsupported datasets {accepted_datasets}"
        )

    # Read dataset and embedding files
    cleaned_dataset = args.dataset + ".clean"
    dataset = read_file(args.cleaned_dir, cleaned_dataset)
    train_test_info = read_file(args.train_test_info_dir, args.dataset)

    return dataset, train_test_info


def get_word_nodes(dataset):
    word_nodes = set()
    for doc in dataset:
        doc_words = doc.split()
        word_nodes.update(doc_words)
    word_nodes = list(word_nodes)

    return word_nodes


def build_bipartite_graph(dataset):
    word_nodes = get_word_nodes(dataset)
    word_id_map = {word:i for i, word in enumerate(word_nodes)}

    G = nx.Graph()
    for doc_id, doc in enumerate(dataset):
        doc_words = doc.split()
        doc_id = "doc_" + str(doc_id)
        G.add_node(doc_id, bipartite=0)
        for word in doc_words:
            word_id = word_id_map[word]
            G.add_node(word_id, bipartite=1)
            G.add_edge(doc_id, word_id)

    return G


def project_bipartite_graph(G, weight_type, threshold):
    doc_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}

    match weight_type:
        case "shared":
            return modified_weighted_projected_graph(G, doc_nodes, threshold)
        case "collab":
            return modified_collaboration_weighted_projected_graph(G, doc_nodes, threshold)
        case "overlap":
            return modified_overlap_weighted_projected_graph(G, doc_nodes, threshold)
        case _:
            raise Exception("unsupported weight_type. use 'shared', 'collab', or 'overlap'")


def get_glove_embeddings(dataset, args):
    print("generating embeddings using glove")
    embeddings = read_embedding_file(args.embeddings_dir, args.embedding_file)

    word_nodes = get_word_nodes(dataset)
    oov_filename = os.path.join(args.embeddings_dir, args.dataset + ".oov")
    print(exists(oov_filename))
    if exists(oov_filename):
        print("loading cached oov embeddings...")
        with open(oov_filename, "rb") as f:
            oov = pkl.load(f)
    else:
        print("missing oov file. building from scratch...")
        oov = create_random_features(word_nodes)
        with open(oov_filename, "wb") as f:
            pkl.dump(oov, f)

    for key, value in oov.items():
        if key not in embeddings:
            embeddings[key] = value

    doc_words = []
    for doc in dataset:
        words = set(doc.split())
        doc_words.append(list(words))

    node_features = []
    for words in doc_words:
        embs = [embeddings[word] for word in words]
        node_features.append(embs)

    # average word embeddings for each doc in the dataset.
    # doc x words x x_dim becomes doc x x_dim
    node_features = list(map(lambda x: np.mean(x, axis=0), node_features))
    return node_features

def get_doc_features(dataset, args):
    node_features = []
    glove_embeddings = get_glove_embeddings(dataset, args)
    node_features.append(glove_embeddings)

    if args.transformer_embeddings:
        model_list = read_file("", "transformer_list")
        for model_name in model_list:
            print(f"generating embeddings using {model_name.strip()}")
            model_name = model_name.strip()
            model = SentenceTransformer(model_name)

            embs = []
            for doc in dataset:
                doc_emb = model.encode(doc)
                embs.append(doc_emb)
            node_features.append(embs)
            print("embeddings generated")

    return node_features


def build_graph(dataset, y_map, doc_name, args):
    G = build_bipartite_graph(dataset)

    # graph projection
    print(f"projecting bipartite graph.")
    start = time.time()
    G = project_bipartite_graph(G, args.weight_type, args.threshold)
    end = time.time()

    print(f"projection time: {(end-start)/60:.2f} minutes")
    print(f"projected graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

    # Create edge index to use in torch geometric
    graph_dict = {}
    graph_dict["edge_index"], graph_dict["edge_weight"] = networkx_to_edge_index(G)

    # Create random features for each node in the graph
    node_features = get_doc_features(dataset, args)
    graph_dict["x"] = node_features

    y = []
    for i in range(len(doc_name)):
        doc_meta = doc_name[i].split('\t')
        label = doc_meta[2]
        y.append(y_map[label])
    graph_dict["y"] = y

    return graph_dict


def run(args):
    cur_graph_dir = os.path.join(args.graphs_dir, args.dataset)
    Path(cur_graph_dir).mkdir(parents=True, exist_ok=True)

    if args.weight_type == None:
        raise Exception(
            "please set the arg weight_type in order to build the graph"
        )

    dataset, train_test_info = get_data(args)

    # check if splits are cached
    cur_graph_dir = f"{args.graphs_dir}{args.dataset}/{args.dataset}"
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
    y = []
    for doc_meta in shuffled_doc_name_list:
        temp = doc_meta.split("\t")
        y.append(temp[2])
    y_map = {label:i for i, label in enumerate(set(y))}

    train_size = len(train_ids)
    val_size = int(args.val_split * train_size)
    real_train_size = train_size - val_size

    ids = train_ids + test_ids
    masks_train = ids[0:real_train_size]
    masks_val = ids[real_train_size:real_train_size+val_size]
    masks_test = ids[train_size:]

    graph_dict = build_graph(shuffled_dataset, y_map, shuffled_doc_name_list, args)
    graph_dict["masks"] = [masks_train, masks_val, masks_test]

    print(f"saving {args.dataset}'s graph information in the following dir: {args.graphs_dir}")
    save_graph(graph_dict, args)


if __name__ == "__main__":
    set_seed(42)
    args = args_build_graph()
    print(args)
    run(args)
