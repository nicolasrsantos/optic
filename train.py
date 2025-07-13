import numpy as np
import time
import torch
import numpy as np
import wandb
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch.nn import functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import APPNP

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
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        if args.gpu_id is None:
            args.gpu_id = 0
        device = torch.device(f"cuda:{args.gpu_id}")
    print(f"running experiment on device {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float('inf')
    early_stopping_count = 0
    best_model_str = args.models_dir + args.dataset + "_" + args.model + ".pt"

    for epoch in range(1, args.n_epochs + 1):
        if early_stopping_count == args.patience:
            print(f"early stopping on epoch {epoch}")
            break

        loss, train_acc, train_f1 = train(train_loader, model, optimizer, device, args)
        val_loss, val_acc, val_f1 = eval(val_loader, model, device, args)

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"epoch {epoch}\tloss {loss:.4f}\tacc {train_acc:.4f}\tf1 {train_f1:.4f}"
                f"\tval_loss {val_loss:.4f}\tval_acc {val_acc:.4f}\tval_f1 {val_f1:.4f}"
            )

        if val_loss <= best_val_loss:
            early_stopping_count = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_str)
        else:
            early_stopping_count += 1

    model.load_state_dict(torch.load(best_model_str))
    test_loss, test_acc, test_f1 = eval(test_loader, model, device, args)

    return test_loss, test_acc, test_f1


def run_exp(args):
    if args.weight_type == None:
        raise Exception("weight_type not specified. please make sure you set --weight_type")
    if args.threshold == None:
        raise ValueError("threshold not set. please make sure you set --threshold")

    args = load_configs(args)
    data_all = get_pyg_graph(args)

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
            model = WeightedGraphSAGE(args.input_dim, args.output_dim, args.hidden_dim, args)
        case "gat":
            model = GAT(args.input_dim, args.output_dim, args.hidden_dim, args.gat_heads)
        case "appnp":
            print("appnp")
            model = MyAPPNP(args.input_dim, args.hidden_dim, args.output_dim, K=2)
        case "chebynet":
            print("chebynet")
            model = MyChebyNet(args.input_dim, args.hidden_dim, args.output_dim, 2)
        case "sgc":
            print("sgc")
            model = MySGC(args.input_dim, args.output_dim, 2)
        case _:
            raise Exception("unsupported model. only gcn, gat, appnp, and graphsage supported.")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    test_loss, test_accuracy, test_f1 = exp(model, args, train_loader, val_loader, test_loader)
    print(f"{test_loss}, {test_accuracy}, {test_f1}")

if __name__ == "__main__":
    set_seed(42)
    args = args_train()
    print(args)

    run_exp(args)
