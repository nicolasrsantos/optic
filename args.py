from argparse import ArgumentParser, BooleanOptionalAction

def args_train():
    parser = ArgumentParser()

    # model and dataset args
    parser.add_argument("--dataset", dest="dataset", type=str,
                        help="use R8, R52, ohsumed or 20ng")
    parser.add_argument("--weight_type", dest="weight_type", type=str,
                        help="set which weight was used by the projection function.\n"
                        "use 'shared', 'collab', or 'overlap'.\n")
    parser.add_argument("--threshold", dest="threshold", type=int)
    parser.add_argument("--model", dest="model", type=str,
                        help="GCN, GAT or SAGE")
    parser.add_argument("--input_dim", dest="input_dim", type=int)
    parser.add_argument("--hidden_dim", dest="hidden_dim", type=int)
    parser.add_argument("--output_dim", dest="output_dim", type=int)
    parser.add_argument("--gat_heads", dest="gat_heads", type=int,
                        help="number of heads for GAT")
    parser.add_argument("--emb_method", dest="emb_method", type=str)
    parser.add_argument("--sample_size", dest="sample_size", type=int,
                        help="neighborhood sampling size per layer. set it to "
                        "-1 to disable neighborhood sampling")

    # training args
    parser.add_argument("--batch_size", dest="batch_size", type=int)
    parser.add_argument("--gpu", dest="cuda", action="store_true",
                        help="whether to train using gpu")
    parser.add_argument("--cpu", dest="cuda", action="store_false",
                        help="whether to train using cpu")
    parser.add_argument("--n_epochs", dest="n_epochs", type=int)
    parser.add_argument("--lr", dest="lr", type=float)
    parser.add_argument("--epoch_log", dest="epoch_log", type=int)
    parser.add_argument("--weight_decay", dest="weight_decay", type=float)
    parser.add_argument("--dropout", dest="dropout", type=float)
    parser.add_argument("--n_runs", dest="n_runs", type=int,
                        help="number of times the network is trained before"
                        "to compute the average metrics on the test set")
    parser.add_argument("--patience", dest="patience", type=int,
                        help="number of epochs without validation loss improvement before early stopping")
    # dir args
    parser.add_argument("--graphs_dir", dest="graphs_dir", type=str,
                        help="directory to save the edge_index, edge_weight, x, y, and masks")
    parser.add_argument("--models_dir", dest="models_dir", type=str,
                        help="dir to store saved models")
    parser.add_argument("--train_test_info_dir", dest="train_test_info_dir", type=str,
                        help="directory to get class information about the docs")
    parser.add_argument("--linear", dest="linear", type=int)
    parser.add_argument("--weighted", dest="weighted", type=int)
    parser.add_argument("--gpu_id", dest="gpu_id", type=int)

    parser.set_defaults(
        dataset="R8", model="GCN", weight_decay=0, batch_size=32, hidden_dim=256, gat_heads=4,
        cuda=True, n_epochs=200, lr=1e-3, epoch_log=5, patience=30, dropout=0.0, gpu_id=0,
        emb_method="glove", n_runs=10, models_dir="models/", sample_size=-1,
        graphs_dir="data/graphs/", train_test_info_dir="data/train_test_info/"
    )

    args = parser.parse_args()
    return args

def args_build_graph():
    parser = ArgumentParser()

    # dataset args
    parser.add_argument("--dataset", dest="dataset", type=str)
    parser.add_argument("--weight_type", dest="weight_type", type=str,
                        help="set which weight is used by the projection function.\n"
                        "use 'shared', 'collab', or 'overlap'.\n"
                        "check networkx's bipartite algorithms documentation for more information.")
    parser.add_argument("--transformer_embeddings", dest="transformer_embeddings", action=BooleanOptionalAction,
                        help="whether to also use sentence transformer to generate embeddings")
    parser.add_argument("--embedding_file", dest="embedding_file", type=str,
                        help="usage example 'glove.6B.300d' for glove's 300 dim 6B file. "
                        "please make sure to set feature_dim with the same dimension of the embedding")
    parser.add_argument("--val_split", dest="val_split", type=float)
    parser.add_argument("--threshold", dest="threshold", type=int)

    # dir args
    parser.add_argument("--cleaned_dir", dest="cleaned_dir", type=str,
                        help="directory of the cleaned data files")
    parser.add_argument("--train_test_info_dir", dest="train_test_info_dir", type=str,
                        help="directory to get class information about the docs")
    parser.add_argument("--corpus_dir", dest="corpus_dir", type=str,
                        help="directory where the raw corpora files are stored")
    parser.add_argument("--embeddings_dir", dest="embeddings_dir", type=str,
                        help="directory where the embedding files are stored")

    parser.set_defaults(
        dataset="R8", transformer_embeddings=True, weight_type="shared",
        val_split=0.1, embedding_file="glove.6B.300d", cleaned_dir="data/cleaned/",
        embeddings_dir="embeddings/", train_test_info_dir="data/train_test_info/",
        graphs_dir="data/graphs/", corpus_dir="data/corpus/"
    )

    args = parser.parse_args()
    return args
