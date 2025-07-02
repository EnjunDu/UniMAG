import os
import torch
import numpy as np
import argparse
import dgl
from tqdm import tqdm


def build_graph_dataset(path, output_filename):
    # 边
    edge_index = torch.load(os.path.join(path, "nc_edges-nodeid.pt")) 
    edge_tensor = torch.tensor(edge_index, dtype=torch.long)  # shape = (num_edges, 2)
    src = edge_tensor[:, 0]
    dst = edge_tensor[:, 1]

    num_nodes = max(src.max().item(), dst.max().item()) + 1
    print(f"num_nodes = {num_nodes}, num_edges = {src.shape[0]}")

    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    # Load labels
    labels_path = os.path.join(path, "labels-w-missing.pt")
    labels = torch.load(labels_path)
    
    # 转为 tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    # 验证长度是否匹配节点数
    if labels_tensor.shape[0] != num_nodes:
        raise ValueError(f"Label length {labels_tensor.shape[0]} does not match number of nodes {num_nodes}")

    graph.ndata["label"] = labels_tensor



    # Load split
    split = torch.load(os.path.join(path, "split.pt"))  # keys: train_idx, val_idx, test_idx
    for key in ["train", "val", "test"]:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        idx = split[f"{key}_idx"]
        mask[idx] = True
        graph.ndata[f"{key}_mask"] = mask

    # Save graph
    save_path = os.path.join(path, output_filename)
    dgl.save_graphs(save_path, [graph])
    print(f"[✔] Saved graph to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to dataset folder (e.g. books-nc)")
    args = parser.parse_args()

    dataset_name = os.path.basename(os.path.normpath(args.path))
    graph_filename = dataset_name + "Graph.pt"

    print(f"[→] Processing dataset: {dataset_name}")

    build_graph_dataset(args.path, graph_filename)

    print("[✓] All files converted successfully.")

if __name__ == "__main__":
    main()
