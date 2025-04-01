import numpy as np
import random
import pickle as pkl
from pathlib import Path
import umap
import matplotlib.pyplot as plt
from os.path import exists
import requests, zipfile, io
import networkx as nx

import torch
from torch_geometric.data import Data

### Auxiliary classes ###
class ProjectedBipartiteGraph(Data):
    def __init__(self, graph_dict):
        super().__init__()
        self.edge_index = graph_dict["edge_index"]
        self.x = graph_dict["x"]
        self.y = graph_dict["y"]
        self.edge_weight = graph_dict["edge_weight"]
        self.train_mask = graph_dict["masks"][0]
        self.val_mask = graph_dict["masks"][1]
        self.test_mask = graph_dict["masks"][2]

    @property
    def num_nodes(self):
        return len(self.y)

    @property
    def num_classes(self):
        return len(list(set(self.y.numpy())))

### Auxiliary functions ###
def read_file(file_dir: str, filename: str) -> list:
    file_content = []
    with open(file_dir + filename + ".txt", "r") as f:
        for line in f.readlines():
            file_content.append(line)

    return file_content


def download_emb_file(emb_dir):
    print("downloading embedding file...")
    zip_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    request = requests.get(zip_url)
    emb_file = zipfile.ZipFile(io.BytesIO(request.content))
    emb_file.extract("glove.6B.300d.txt", path=emb_dir)
    print("embedding file downloaded.")


def read_embedding_file(emb_dir, emb_file):
    if not exists(emb_dir + emb_file + ".txt"):
        download_emb_file(emb_dir)

    embeddings = {}
    with open(emb_dir + emb_file + ".txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            word = data[0]
            emb_vec = [float(i) for i in data[1:]]
            embeddings[word] = emb_vec

    return embeddings


def create_random_features(data, low = -0.01, high = 0.01, x_dim = 300):
    random_features = {}

    if isinstance(data, dict):
        data = list(data.keys())
    elif isinstance(data, set):
        data = list(data)

    for i in range(len(data)):
        random_features[data[i]] = np.random.uniform(low, high, x_dim)

    return random_features


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def networkx_to_edge_index(G):
    source = []
    target = []
    edge_weight = []

    for edge in G.edges(data=True):
        i, j = edge[0].split("_"), edge[1].split("_")
        i, j = int(i[1]), int(j[1])

        source.append(i)
        target.append(j)
        source.append(j)
        target.append(i)

        edge_weight.append(edge[2]["weight"]) # (i, j)
        edge_weight.append(edge[2]["weight"]) # (j, i)
    edge_index = [source, target]

    return edge_index, edge_weight


def networkx_to_node_list(projected_nodes):
    nodes = set()
    for node in projected_nodes:
        node = node.split("_")
        node = int(node[1])
        nodes.add(node)

    return nodes


def check_graph_properties(G):
    print("checking graph properties")
    print(f"\tGraph has isolated nodes: {G.has_isolated_nodes()}")
    print(f"\tGraph has self loops: {G.has_self_loops()}")
    print(f"\tGraph is undirected: {G.is_undirected()}")


def read_graph(args):
    read_dir = f"{args.graphs_dir}{args.dataset}/threshold_{args.threshold}/"
    read_str = read_dir + f"{args.dataset}_thr{args.threshold}_{args.weight_type}"

    if "paraphrase" in args.emb_method:
        emb_method = "sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2"
    elif "intfloat" in args.emb_method:
        emb_method = "intfloat-multilingual-e5-large"
    elif "glove" in args.emb_method:
        emb_method = "glove"
    else:
        raise Exception(f"emb method not supported. expected 'paraphrase', 'intfloat' or 'glove' (got {args.emb_method})")

    graph_dict = {}
    with open(read_str + "_" + emb_method + ".x_feat", "rb") as f:
        graph_dict["x"] = pkl.load(f)
    with open(read_str + ".edge_index", "rb") as f:
        graph_dict["edge_index"] = pkl.load(f)
    with open(read_str + ".y", "rb") as f:
        graph_dict["y"] = pkl.load(f)
    with open(read_str + ".edge_weight", "rb") as f:
        graph_dict["edge_weight"] = pkl.load(f)
    with open(read_str + ".masks", "rb") as f:
        graph_dict["masks"] = pkl.load(f)

    return graph_dict


def save_graph(graph_dict, args):
    save_dir = f"{args.graphs_dir}{args.dataset}/threshold_{args.threshold}/"
    save_str = save_dir + f"{args.dataset}_thr{args.threshold}_{args.weight_type}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    with open(save_str + ".edge_index", "wb") as f:
        pkl.dump(graph_dict["edge_index"], f)
    with open(save_str + ".edge_weight", "wb") as f:
        pkl.dump(graph_dict["edge_weight"], f)
    with open(save_str + ".y", "wb") as f:
        pkl.dump(graph_dict["y"], f)
    with open(save_str + ".masks", "wb") as f:
        pkl.dump(graph_dict["masks"], f)
    with open(save_str + "_glove.x_feat", "wb") as f:
        pkl.dump(graph_dict["x"][0], f)

    if args.transformer_embeddings:
        transformer_list = read_file("", "transformer_list")
        for i in range(len(transformer_list)):
            embs = graph_dict["x"][i+1] # +1 because 0 is contains glove embeddings
            model_name = transformer_list[i]

            weights_filename = save_str + "_" + model_name.strip().replace("/", "-")
            with open(weights_filename + ".x_feat", "wb") as f:
                pkl.dump(embs, f)

    print("graph information saved")


def prepare_graph(graph_dict):
    ret_dict = {}
    ret_dict["edge_index"] = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    ret_dict["edge_weight"] = torch.tensor(graph_dict["edge_weight"], dtype=torch.float)

    x = np.array(graph_dict["x"])
    ret_dict["x"] = torch.tensor(x, dtype=torch.float)
    ret_dict["y"] = torch.tensor(graph_dict["y"], dtype=torch.long)

    n_nodes = len(ret_dict["y"])
    train_idx, val_idx, test_idx = graph_dict["masks"][0], graph_dict["masks"][1], graph_dict["masks"][2]
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = 1
    val_mask[val_idx] = 1
    test_mask[test_idx] = 1
    ret_dict["masks"] = [train_mask, val_mask, test_mask]

    return ret_dict


def plot_embeddings(embeddings, predicted):
    reducer = umap.UMAP()
    data = reducer.fit_transform(embeddings)

    plt.scatter(data[:, 0], data[:, 1], c=predicted, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()


def to_csv(embedding_model, loss, acc, f1, is_avg, args):
    csv = f"{args.model},{embedding_model},{args.weight_type},{args.lr},"
    csv = csv + f"{args.batch_size},{args.threshold},{args.weight_decay},"
    csv = csv + f"{args.dropout},{args.input_dim},{args.hidden_dim},"
    csv = csv + f"{args.sample_size},{args.gat_heads},{loss:.4f},{acc:.4f},{f1:.4f}\n"

    csv_file = f"results/{args.dataset}_results.csv"
    if is_avg:
        csv_file = f"results/{args.dataset}_avg_results.csv"

    with open(csv_file, "a") as f:
        f.write(csv)


def get_pyg_graph(args):
    graph_dict = read_graph(args)
    graph_dict = prepare_graph(graph_dict)
    return ProjectedBipartiteGraph(graph_dict)


def modified_weighted_projected_graph(B, nodes, threshold, ratio=False):
    # Code extracted from the Networkx library.
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n, B.nodes[n]) for n in nodes)
    n_top = len(B) - len(nodes)

    if n_top < 1:
        raise nx.exception.NetworkXAlgorithmError(
            f"the size of the nodes to project onto ({len(nodes)}) is >= the graph size ({len(B)}).\n"
            "They are either not a valid bipartite partition or contain duplicates"
        )

    for u in nodes:
        unbrs = set(B[u])
        nbrs2 = {n for nbr in unbrs for n in B[nbr]} - {u}
        for v in nbrs2:
            vnbrs = set(pred[v])
            common = unbrs & vnbrs
            if len(common) >= threshold:
                if not ratio:
                    weight = len(common)
                else:
                    weight = len(common) / n_top
                G.add_edge(u, v, weight=weight)
    return G


def modified_collaboration_weighted_projected_graph(B, nodes, threshold):
    # Code extracted from the Networkx library.
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n, B.nodes[n]) for n in nodes)
    common_max = 0
    common_, cnt = 0, 0
    for u in nodes:
        unbrs = set(B[u])
        nbrs2 = {n for nbr in unbrs for n in B[nbr] if n != u}
        for v in nbrs2:
            vnbrs = set(pred[v])
            common = unbrs & vnbrs
            if len(common) > common_max:
                common_max = len(common)
            common_ += len(common)
            cnt += 1
            if len(common) >= threshold:
                common_degree = (len(B[n]) for n in unbrs & vnbrs)
                weight = sum(1.0 / (deg - 1) for deg in common_degree if deg > 1)
                G.add_edge(u, v, weight=weight)
    return G


def modified_overlap_weighted_projected_graph(B, nodes, threshold, jaccard=True):
    # Code extracted from the Networkx library.
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n, B.nodes[n]) for n in nodes)
    for u in nodes:
        unbrs = set(B[u])
        nbrs2 = {n for nbr in unbrs for n in B[nbr]} - {u}
        for v in nbrs2:
            vnbrs = set(pred[v])
            common = unbrs & vnbrs
            if len(common) >= threshold:
                if jaccard:
                    wt = len(unbrs & vnbrs) / len(unbrs | vnbrs)
                else:
                    wt = len(unbrs & vnbrs) / min(len(unbrs), len(vnbrs))
                G.add_edge(u, v, weight=wt)
    return G


def load_configs(args):
    if "paraphrase" in args.emb_method:
        args.input_dim = 384
    elif "roberta" in args.emb_method or "intfloat" in args.emb_method:
        args.input_dim = 1024
    else:
        args.input_dim = 300

    if args.dataset == "R8":
        args.output_dim = 8
    elif args.dataset == "R52":
        args.output_dim = 52
    elif args.dataset == "ohsumed":
        args.output_dim = 23
    elif args.dataset == "mr":
        args.output_dim = 2
    elif args.dataset == "TREC":
        args.output_dim = 6
    elif args.dataset == "WebKB":
        args.output_dim = 8
    elif args.dataset == "SST1":
        args.output_dim = 5
    elif args.dataset == "SST2":
        args.output_dim = 2
    elif args.dataset == "20ng":
        args.output_dim = 20

    return args