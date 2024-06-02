import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
import time

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return F.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data(batch_size=1, path='', suffix='', online=True):

    loc_train = np.load(path + '/loc_train' + suffix + '.npy')#.transpose((0,2,3,1))
    vel_train = np.load(path + '/vel_train' + suffix + '.npy')#.transpose((0,2,3,1))
    edges_train = np.load(path + '/edges_train' + suffix + '.npy')

    #print("edges_train: ", edges_train.shape, edges_train[0])

    if online == True:
        loc_valid, loc_test = loc_train, loc_train
        vel_valid, vel_test = vel_train, vel_train
        edges_valid, edges_test = edges_train, edges_train
    else:
        loc_valid = np.load(path + '/loc_valid' + suffix + '.npy')
        vel_valid = np.load(path + '/vel_valid' + suffix + '.npy')
        edges_valid = np.load(path + '/edges_valid' + suffix + '.npy')

        loc_test = np.load(path + '/loc_test' + suffix + '.npy')
        vel_test = np.load(path + '/vel_test' + suffix + '.npy')
        edges_test = np.load(path + '/edges_test' + suffix + '.npy')

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64) # if there is interaction -> 1, else -> 0

    #print("edges_train: ", edges_train.shape, edges_train[0]) # (num sims=15000, num nodes=5, num nodes=5) -> (15000, 25) -> (15000, 20)
    #input("here")

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    if online == True:
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    else:
        train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min


def load_kuramoto_data(batch_size=1, suffix=''):
    feat_train = np.load('data/feat_train' + suffix + '.npy')
    edges_train = np.load('data/edges_train' + suffix + '.npy')
    feat_valid = np.load('data/feat_valid' + suffix + '.npy')
    edges_valid = np.load('data/edges_valid' + suffix + '.npy')
    feat_test = np.load('data/feat_test' + suffix + '.npy')
    edges_test = np.load('data/edges_test' + suffix + '.npy')

    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]

    # Normalize each feature dim. individually
    feat_max = feat_train.max(0).max(0).max(0)
    feat_min = feat_train.min(0).min(0).min(0)

    feat_max = np.expand_dims(np.expand_dims(np.expand_dims(feat_max, 0), 0), 0)
    feat_min = np.expand_dims(np.expand_dims(np.expand_dims(feat_min, 0), 0), 0)

    # Normalize to [-1, 1]
    feat_train = (feat_train - feat_min) * 2 / (feat_max - feat_min) - 1
    feat_valid = (feat_valid - feat_min) * 2 / (feat_max - feat_min) - 1
    feat_test = (feat_test - feat_min) * 2 / (feat_max - feat_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_kuramoto_data_old(batch_size=1, suffix=''):
    feat_train = np.load('data/old_kuramoto/feat_train' + suffix + '.npy')
    edges_train = np.load('data/old_kuramoto/edges_train' + suffix + '.npy')
    feat_valid = np.load('data/old_kuramoto/feat_valid' + suffix + '.npy')
    edges_valid = np.load('data/old_kuramoto/edges_valid' + suffix + '.npy')
    feat_test = np.load('data/old_kuramoto/feat_test' + suffix + '.npy')
    edges_test = np.load('data/old_kuramoto/edges_test' + suffix + '.npy')

    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_motion_data(batch_size=1, suffix=''):
    feat_train = np.load('data/motion_train' + suffix + '.npy')
    feat_valid = np.load('data/motion_valid' + suffix + '.npy')
    feat_test = np.load('data/motion_test' + suffix + '.npy')
    adj = np.load('data/motion_adj' + suffix + '.npy')

    # NOTE: Already normalized

    # [num_samples, num_nodes, num_timesteps, num_dims]
    num_nodes = feat_train.shape[1]

    edges_train = np.repeat(np.expand_dims(adj.flatten(), 0),
                            feat_train.shape[0], axis=0)
    edges_valid = np.repeat(np.expand_dims(adj.flatten(), 0),
                            feat_valid.shape[0], axis=0)
    edges_test = np.repeat(np.expand_dims(adj.flatten(), 0),
                           feat_test.shape[0], axis=0)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(np.array(edges_train, dtype=np.int64))
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(np.array(edges_valid, dtype=np.int64))
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(np.array(edges_test, dtype=np.int64))

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
        [num_nodes, num_nodes])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def get_minimum_distance(data):
    data = data[:, :, :, :2].transpose(1, 2)
    data_norm = (data ** 2).sum(-1, keepdim=True)
    dist = data_norm + \
           data_norm.transpose(2, 3) - \
           2 * torch.matmul(data, data.transpose(2, 3))
    min_dist, _ = dist.min(1)
    return min_dist.view(min_dist.size(0), -1)


def get_buckets(dist, num_buckets):
    dist = dist.cpu().data.numpy()

    min_dist = np.min(dist)
    max_dist = np.max(dist)
    bucket_size = (max_dist - min_dist) / num_buckets
    thresholds = bucket_size * np.arange(num_buckets)

    bucket_idx = []
    for i in range(num_buckets):
        if i < num_buckets - 1:
            idx = np.where(np.all(np.vstack((dist > thresholds[i],
                                             dist <= thresholds[i + 1])), 0))[0]
        else:
            idx = np.where(dist > thresholds[i])[0]
        bucket_idx.append(idx)

    return bucket_idx, thresholds


def get_correct_per_bucket(bucket_idx, pred, target):
    pred = pred.cpu().numpy()[:, 0]
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def get_correct_per_bucket_(bucket_idx, pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False, delta_vel = None):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const

    if delta_vel != None:
        neg_log_p = neg_log_p * delta_vel

    return neg_log_p.sum() / (target.size(0) * target.size(1))


def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float64(correct) / (target.size(0) * target.size(1))


def plot_trajectories(preds, target, path):
    # preds: (num_atoms,timesteps,num_features) output: (num_atoms,timesteps,num_features)
    num_atoms = preds.shape[0]
    timesteps = preds.shape[1]

    #colors = np.zeros((num_atoms, timesteps))
    #for i in range(num_atoms):
    #    colors[i,:] += (i+1) / num_atoms

    colors = np.ones((num_atoms, timesteps))
    for i in range(num_atoms):
        colors[i,:] += i

    alphas = np.ones((num_atoms, timesteps))
    for i in range(timesteps):
        alphas[:,i] = (i+1) / timesteps

    f, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].scatter(x=preds[:,:,0].detach().cpu(), y=preds[:,:,1].detach().cpu(), c=colors, alpha=1)#alphas)
    ax[1].scatter(x=target[:,:,0].detach().cpu(), y=target[:,:,1].detach().cpu(), c=colors, alpha=1)#alphas)
    plt.show()
    plt.savefig(path)
    plt.close()


#### NOTE: CODE FROM DNRI PAPER

# Code from NRI.
def normalize(data, data_max, data_min):
	return (data - data_min) * 2 / (data_max - data_min) - 1


def unnormalize(data, data_max, data_min):
	return (data + 1) * (data_max - data_min) / 2. + data_min


def get_edge_inds(num_vars):
	edges = []
	for i in range(num_vars):
		for j in range(num_vars):
			if i == j:
				continue
			edges.append([i, j])
	return edges

class CmuMotionData(Dataset):
    def __init__(self, name, data_path, mode, params, test_full=False, mask_ind_file=None):
        self.name = name
        self.data_path = data_path
        self.mode = mode
        self.params = params
        self.train_data_len = params.get('train_data_len', -1)
        # Get preprocessing stats.
        loc_max, loc_min, vel_max, vel_min = self._get_normalize_stats()
        self.loc_max = loc_max
        self.loc_min = loc_min
        self.vel_max = vel_max
        self.vel_min = vel_min
        self.test_full = test_full

        # Load data.
        self._load_data()
        self.expand_train = params.get('expand_train', False)
        if self.mode == 'train' and self.expand_train and self.train_data_len > 0:
            self.all_inds = []
            for ind in range(len(self.feat)):
                t_ind = 0
                while t_ind < len(self.feat[ind]):
                    self.all_inds.append((ind, t_ind))
                    t_ind += self.train_data_len
        else:
            self.expand_train = False

    def __getitem__(self, index):
        if self.expand_train:
            ind, t_ind = self.all_inds[index]
            start_ind = np.random.randint(t_ind, t_ind + self.train_data_len)

            feat = self.feat[ind][start_ind:start_ind + self.train_data_len]
            if len(feat) < self.train_data_len:
                feat = self.feat[ind][-self.train_data_len:]
            return {'inputs':feat}
        else: 
            inputs = self.feat[index]
            size = len(inputs)
            if self.mode == 'train' and self.train_data_len > 0 and size > self.train_data_len:
                start_ind = np.random.randint(0, size-self.train_data_len)
                inputs = inputs[start_ind:start_ind+self.train_data_len]

            result = (inputs, torch.zeros_like(inputs))
        return result

    def __len__(self, ):
        if self.expand_train:
            return len(self.all_inds)
        else:
            return len(self.feat)

    def _get_normalize_stats(self,):
        train_loc = np.load(self._get_npy_path('loc', 'train'), allow_pickle=True)
        train_vel = np.load(self._get_npy_path('vel', 'train'), allow_pickle=True)
        try:
            train_loc.max()
            self.dynamic_len = False
        except:
            self.dynamic_len = True
        if self.dynamic_len:
            max_loc = max(x.max() for x in train_loc)
            min_loc = min(x.min() for x in train_loc)
            max_vel = max(x.max() for x in train_vel)
            min_vel = min(x.min() for x in train_vel)
            return max_loc, min_loc, max_vel, min_vel
        else:
            return train_loc.max(), train_loc.min(), train_vel.max(), train_vel.min()

    def _load_data(self, ):
        #print('***Experiment hack: evaling on training.***')
        # Load data
        self.loc_feat = np.load(self._get_npy_path('loc', self.mode), allow_pickle=True)
        self.vel_feat = np.load(self._get_npy_path('vel', self.mode), allow_pickle=True)
        #self.edge_feat = np.load(self._get_npy_path('edges', self.mode))

        # Perform preprocessing.
        if self.dynamic_len:
            self.loc_feat = [normalize(feat, self.loc_max, self.loc_min) for feat in self.loc_feat]
            self.vel_feat = [normalize(feat, self.vel_max, self.vel_min) for feat in self.vel_feat]
            self.feat = [np.concatenate([loc_feat, vel_feat], axis=-1) for loc_feat, vel_feat in zip(self.loc_feat, self.vel_feat)]
            self.feat = [torch.from_numpy(np.array(feat, dtype=np.float32)) for feat in self.feat]
            print("FEATURE LEN: ",len(self.feat))
        else:
            self.loc_feat = normalize(
                self.loc_feat, self.loc_max, self.loc_min)
            self.vel_feat = normalize(
                self.vel_feat, self.vel_max, self.vel_min)

            # Reshape [num_sims, num_timesteps, num_agents, num_dims]
            #self.loc_feat = np.transpose(self.loc_feat, [0, 1, 3, 2])
            #self.vel_feat = np.transpose(self.vel_feat, [0, 1, 3, 2])
            self.feat = np.concatenate([self.loc_feat, self.vel_feat], axis=-1)

            # Convert to pytorch cuda tensor.
            self.feat = torch.from_numpy(
                np.array(self.feat, dtype=np.float32))  # .cuda()

            # Only extract the first 49 frame if testing.
            if self.mode == 'test' and not self.test_full:
                self.feat = self.feat[:, :49]

    def _get_npy_path(self, feat, mode):
        return '%s/%s_%s_%s.npy' % (self.data_path,
                                    feat,
                                    mode,
                                    self.name)

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
            if edges[0,30*idx1+idx2,-1] > 0.3:
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
