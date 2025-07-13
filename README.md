This repository is the official PyTorch implementation of "One-mode Projection of Bipartite Graphs for Text Classification using Graph Neural Networks" published in the 40th ACM/SIGAPP Symposium On Applied Computing (SAC) 2025.

## Requirements

This code was implemented using Python 3.11.5, CUDA 12.2 and the following packages:

- `networkx==3.3`
- `nltk==3.8.1`
- `numpy==1.26.4`
- `python_igraph==0.11.5`
- `PyYAML==6.0.1`
- `scikit_learn==1.4.2`
- `scipy==1.13.1`
- `torch==2.1.2`
- `torch_geometric==2.5.3`
- `python_igraph==0.11.5`

## How to run the code

### Build Graph
In order to use our method, first build the graph you wanna perform text classification using the following command:

    $ python build_graph.py --dataset <dataset_name> --threshold <threshold_for_edge_prunning>

More info about the edge prunning is available in Table 2 of our paper.

The following arguments allow the modification of the script's parameters:

- `--weight_type`

    Defines what is the edge weighing strategy used to build the graph. The options are 'shared', 'overlap', and 'collab' for number of shared neighbors, jaccard index, and newman's collaboration model.
  
    Default: `shared`

- `--transformer_embeddings`

    Use this flag if you wanna use sentence transformer embeddings instead of GloVe embeddings.
  
    Default: `True`

- `--val_split`

    Defines the percentage of nodes used for validation.

    Default: `0.1`

### Model training
After building the graph, run the following command to train the model:

    $ python train.py --dataset <dataset_name> --weight_type <weight_type_used_to_build_graph> --threshold <threshold_used_to_build_graph>

The following arguments allow the modification of the training parameters:

- `--model`

  Select the GNN used for training.

  Default: `GCN`
  
- `--lr`

    Modifies the model's learning rate.
  
    Default: `1e-3`

- `--batch_size`

    Controls our method's batch size.
  
    Default: `32`

- `--hidden_dim`

    GNN hidden dimension size. 

    Default: `256`

- `--n_epochs`

    Number of training epochs.

    Default: `200`

- `--weight_decay`

    Weight decay used during training.
  
    Default: `0`

- `--dropout`

    Dropout used during training.
  
    Default: `0`

- `--sample_size`

    Neighborhood sampling size per GNN layer. Set it to -11 to disable neighborhood sampling.
  
    Default: `-1`

- `--emb_method`

    Used to inform the script which embedding method was used to build the graph.
  
    Default: `glove`

- `--patience`

    Number of epochs without validation loss before early stopping.

    Default: `30`

- `--epoch_log`

    Prints information about the network's training every <epoch_log> steps.

    Default: `5`

- `--cuda`

    Trains the network using a GPU (if available).

    Default: `true`

## Reference
```
@inbook{10.1145/3672608.3707879,
  author = {Roque dos Santos, N\'{\i}colas and Minatel, Diego and Dem\'{e}trius Baria Valejo, Alan and de Andrade Lopes, Alneu},
  title = {One-mode Projection of Bipartite Graphs for Text Classification using Graph Neural Networks},
  year = {2025},
  isbn = {9798400706295},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  booktitle = {Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing},
  pages = {645–647},
  numpages = {3}
}
```
