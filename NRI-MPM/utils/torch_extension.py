import torch
from torch import Tensor
from itertools import combinations
from utils.general import prod
import numpy as np
import matplotlib.pyplot as plt

def edge_f1(preds, target):
    _, preds = preds.max(-1)
    tp = (preds.t() * target).sum().float()
    tn = ((preds.t() - 1) * (target - 1)).sum().float()
    fp = int(- (preds.t() * (target - 1)).sum().float())
    fn = int(- ((preds.t() - 1) * target).sum().float())
    return (tp, tn, fp, fn)

def edge_accuracy(preds: Tensor, target: Tensor) -> float:
    """
    Compute the accuracy of edge prediction (relation reconstruction).

    Args:
        preds: [E, batch, K], probability distribution of K types of relation for each edge
        target: [batch, E], ground truth relations

    Return:
         accuracy of edge prediction (relation reconstruction)
    """
    _, preds = preds.max(-1)
    correct = (preds.t() == target).sum().float()
    return correct / (target.size(0) * target.size(1))


def asym_rate(x: Tensor, size: int) -> float:
    """
    Given an edge list of a graph, compute the rate of asymmetry.

    Args:
        x: [batch, E], edge indicator
        size: number of nodes of the graph

    Return:
        rate of asymmetry
    """
    # get the edge indicator of a transposed adjacency matrix
    idx = transpose_id(size)
    x_t = x[:, idx]
    rate = (x != x_t).sum().float() / (x.shape[0] * x.shape[1])
    return rate


def transpose_id(size: int) -> Tensor:
    """
    Return the edge list corresponding to a transposed adjacency matrix.
    """
    idx = torch.arange(size * (size - 1))
    ii = idx // (size - 1)
    jj = idx % (size - 1)
    jj = jj * (jj < ii).long() + (jj + 1) * (jj >= ii).long()
    index = jj * (size - 1) + ii * (ii < jj).long() + (ii - 1) * (ii > jj).long()
    return index


def transpose(x: Tensor, size: int) -> Tensor:
    """
    Transpose the edge features x.
    """
    index = transpose_id(size)
    return x[index]


def sym_hard(x: Tensor, size: int) -> Tensor:
    """
    Given the edge features x, set x(e_ji) = x(e_ij) to impose hard symmetric constraints.
    """
    i, j = np.array(list(combinations(range(size), 2))).T
    idx_s = j * (size - 1) + i * (i < j) + (i - 1) * (i > j)
    idx_t = i * (size - 1) + j * (j < i) + (j - 1) * (j > i)
    x[idx_t] = x[idx_s]
    return x


def my_bn(x: Tensor, bn: torch.nn.BatchNorm1d) -> Tensor:
    """
    Applying BatchNorm1d to a multi-dimesional tensor x.
    """
    shape = x.shape
    z = x.view(prod(shape[:-1]), shape[-1])
    z = bn(z)
    z = z.view(*shape)
    return z


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape)
    return - (- (U.relu() + eps).log().clamp(max=0.) + eps).log()


def sample_gumbel_max(logits, eps=1e-10, one_hot=False):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda(logits.device)
    y = logits + gumbel_noise
    ms, index = y.max(-1, keepdim=True)
    es = (y >= ms) if one_hot else index
    return es


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda(logits.device)
    y = logits + gumbel_noise
    return (y / tau).softmax(-1)


from mpl_toolkits import mplot3d
def plot_motions_edge(preds, rel_rec, rel_send, edges, path):
    # preds: (num_atoms,timesteps,num_features) output: (num_atoms,timesteps,num_features)
    num_atoms = preds.shape[0]
    timesteps = preds.shape[1]
    j = timesteps - 1
    # Creating figure
    fig = plt.figure(figsize = (7, 7))
    ax = plt.axes(projection ="3d")

    # Creating plot
    ax.scatter3D(preds[:,j,2].detach().cpu(), preds[:,j,0].detach().cpu(), preds[:,j,1].detach().cpu(), color = "green")
    for idx1 in range(31):
        for idx2 in range(30):
            _, edge_idx1 = torch.max(rel_rec[30*idx1+idx2,:], dim=-1)
            _, edge_idx2 = torch.max(rel_send[30*idx1+idx2,:], dim=-1)
            if edges[0,30*idx1+idx2,-1] > 0.7:
                ax.plot([preds[edge_idx1,j,2].detach().cpu(), preds[edge_idx2,j,2].detach().cpu()], 
                        [preds[edge_idx1,j,0].detach().cpu(), preds[edge_idx2,j,0].detach().cpu()], 
                        [preds[edge_idx1,j,1].detach().cpu(), preds[edge_idx2,j,1].detach().cpu()], 
                        color = "red", linewidth=edges[0,30*idx1+idx2,-1])

    edge = np.load("../data/motion/processed/35/edges.npy")
    for edge_idx in range(edge.shape[0]):
        ax.plot([preds[edge[edge_idx,0],j,2].detach().cpu(), preds[edge[edge_idx,1],j,2].detach().cpu()], 
                [preds[edge[edge_idx,0],j,0].detach().cpu(), preds[edge[edge_idx,1],j,0].detach().cpu()], 
                [preds[edge[edge_idx,0],j,1].detach().cpu(), preds[edge[edge_idx,1],j,1].detach().cpu()], 
                color = "black", linewidth=3)

    # show plot
    ax.set_xlim(-1, 1)  # Set x-axis range
    ax.set_ylim(0, 1)  # Set y-axis range
    ax.set_zlim(-1, 0)  # Set z-axis range

    plt.show()
    plt.savefig(path)
    plt.close()

def plot_motions_edge_static(prob, rel_rec, rel_send, path):
    joint = np.load("../data/motion/processed/35/joint.npy") # (31, 2)
    edge = np.load("../data/motion/processed/35/edges.npy")

    fig = plt.figure(figsize= (7, 7))
    plt.scatter(x=joint[:,0], y=joint[:,1], color = "blue", s=50)
    plt.plot([joint[edge[:,0],0], joint[edge[:,1],0]],
            [joint[edge[:,0],1], joint[edge[:,1],1]], 
            color = "gray", linewidth=4)

    edge_threshold = torch.topk(prob[0,:,-1], 30)[0][-1]
    for idx1 in range(31):
        for idx2 in range(30):
            _, edge_idx1 = torch.max(rel_rec[30*idx1+idx2,:], dim=-1)
            _, edge_idx2 = torch.max(rel_send[30*idx1+idx2,:], dim=-1)
            
            if prob[0,30*idx1+idx2,-1] >= edge_threshold:
                plt.plot([joint[edge_idx1,0], joint[edge_idx2,0]], 
                        [joint[edge_idx1,1], joint[edge_idx2,1]], 
                        color = "red")

    plt.plot(edge)
    plt.xlim(-0.7,1.7)
    plt.ylim(0,0.95)
    plt.show()
    plt.savefig(path)
    plt.close()

def plot_motions(preds, target, path):
    # preds: (num_atoms,timesteps,num_features) output: (num_atoms,timesteps,num_features)
    num_atoms = preds.shape[0]
    timesteps = preds.shape[1]
    edge = np.load("../data/motion/processed/35/edges.npy")
    j = timesteps - 1
    # Creating figure
    fig = plt.figure(figsize = (7, 7))
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(preds[:,j,2].detach().cpu(), preds[:,j,0].detach().cpu(), preds[:,j,1].detach().cpu(), color = "green")
    ax.scatter3D(target[:,j,2].detach().cpu(), target[:,j,0].detach().cpu(), target[:,j,1].detach().cpu(), color = "blue")
    for edge_idx in range(edge.shape[0]):
        ax.plot([preds[edge[edge_idx,0],j,2].detach().cpu(), preds[edge[edge_idx,1],j,2].detach().cpu()], 
                [preds[edge[edge_idx,0],j,0].detach().cpu(), preds[edge[edge_idx,1],j,0].detach().cpu()], 
                [preds[edge[edge_idx,0],j,1].detach().cpu(), preds[edge[edge_idx,1],j,1].detach().cpu()], 
                color = "black", linewidth=3)
        ax.plot([target[edge[edge_idx,0],j,2].detach().cpu(), target[edge[edge_idx,1],j,2].detach().cpu()], 
                [target[edge[edge_idx,0],j,0].detach().cpu(), target[edge[edge_idx,1],j,0].detach().cpu()], 
                [target[edge[edge_idx,0],j,1].detach().cpu(), target[edge[edge_idx,1],j,1].detach().cpu()], 
                color = "r", linewidth=3)

    # show plot
    ax.set_xlim(-1, 1)  # Set x-axis range
    ax.set_ylim(0, 1)  # Set y-axis range
    ax.set_zlim(-1, 0)  # Set z-axis range

    plt.show()
    plt.savefig(path)
    plt.close()