import numpy as np
import time
import torch
import numpy as np
import wandb
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from copy import deepcopy

import torch
from torch.nn import functional as F
from torch_geometric.loader import NeighborLoader

from utils import *
from args import *
from models import *

Path("models/").mkdir(exist_ok=True, parents=True)
CKPT_PATH = f"models/{time.time_ns()}.pt"

import os
os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]= "10000"

def train(loader, model, optimizer, device, args):
    model.train()

    y_pred, y_true = [], []
    total_examples = total_loss = 0
    for batch in loader:
        optimizer.zero_grad()

        batch = batch.to(device)
        batch_size = batch.batch_size

        out = model(batch.x, batch.edge_index, batch.edge_weight, args)
        loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
        preds = F.softmax(out[:batch_size], dim=-1)
        preds = preds.argmax(dim=-1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(batch.y[:batch_size].cpu().numpy())

        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        train_loss = total_loss / total_examples
    train_acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return train_loss, train_acc, f1


@torch.inference_mode()
def eval(loader, model, device, args):
    model.eval()

    y_pred, y_true = [], []
    total_examples = total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.batch_size

        out = model(batch.x, batch.edge_index, batch.edge_weight, args)
        loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
        preds = F.softmax(out[:batch_size], dim=-1)
        preds = preds.argmax(dim=-1)

        y_pred.extend(preds.cpu().numpy())
        y_true.extend(batch.y[:batch_size].cpu().numpy())

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        eval_loss = total_loss / total_examples
    eval_acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return eval_loss, eval_acc, f1


def exp(model, args, train_loader, val_loader, test_loader):
    wandb.init(
        config={
            "threshold": f"{args.threshold}",
            "dataset": f"{args.dataset}",
            "weight_type": f"{args.weight_type}",
            "model": f"{args.model}",
            "sample_size": f"{args.sample_size}",
            "args": args,
        }
    )

    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        if args.gpu_id is None:
            args.gpu_id = 0
        device = torch.device(f"cuda:{args.gpu_id}")
    print(f"running experiment on device {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_model = deepcopy(model.state_dict())
    best_val_loss = float('inf')
    early_stopping_count = 0
    
    for epoch in range(1, args.n_epochs + 1):
        if early_stopping_count == args.patience:
            print(f"early stopping on epoch {epoch}")
            break

        loss, train_acc, train_f1 = train(train_loader, model, optimizer, device, args)
        val_loss, val_acc, val_f1 = eval(val_loader, model, device, args)

        wandb.log({
            "loss": loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

        if epoch == 1 or epoch % args.epoch_log == 0:
            print(
                f"epoch {epoch}\tloss {loss:.4f}\tacc {train_acc:.4f}\tf1 {train_f1:.4f}"
                f"\tval_loss {val_loss:.4f}\tval_acc {val_acc:.4f}\tval_f1 {val_f1:.4f}"
            )

        if val_loss <= best_val_loss:
            early_stopping_count = 0
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())
        else:
            early_stopping_count += 1

    model.load_state_dict(best_model)
    test_loss, test_acc, test_f1 = eval(test_loader, model, device, args)

    return test_loss, test_acc, test_f1


def run_exp(args):
    if args.weight_type == None:
        raise Exception("weight_type not specified. please make sure you set --weight_type")
    if args.threshold == None:
        raise ValueError("threshold not set. please make sure you set --threshold")

    args = load_configs(args)
    data_all = get_pyg_graph(args)

    check_graph_properties(data_all)

    train_loader = NeighborLoader(
        data_all,
        num_neighbors=[args.sample_size] * 2,
        batch_size=args.batch_size,
        input_nodes=data_all.train_mask
    )
    val_loader = NeighborLoader(
        data_all,
        num_neighbors=[args.sample_size] * 2,
        batch_size=args.batch_size,
        input_nodes=data_all.val_mask
    )
    test_loader = NeighborLoader(
        data_all,
        num_neighbors=[args.sample_size] * 2,
        batch_size=args.batch_size,
        input_nodes=data_all.test_mask
    )

    match args.model.lower():
        case "gcn":
            model = GCN(args.input_dim, args.output_dim, args.hidden_dim, args)
        case "sage":
            #model = GraphSAGE(args.input_dim, args.output_dim, args.hidden_dim, args)
            model = WeightedGraphSAGE(args.input_dim, args.output_dim, args.hidden_dim, args)
        case "gat":
            model = GAT(args.input_dim, args.output_dim, args.hidden_dim, args.gat_heads)
        case _:
            raise Exception("unsupported model. only gcn, gat, and graphsage supported.")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    test_loss, test_accuracy, test_f1 = exp(model, args, train_loader, val_loader, test_loader)
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1
    })
        
    # wandb.finish()
    del data_all, train_loader, val_loader, test_loader


if __name__ == "__main__":
    set_seed(42)
    args = args_train()
    print(args)

    wandb.login()

    run_exp(args)