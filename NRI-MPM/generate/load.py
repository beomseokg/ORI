"""
Re-implementation of data preprocessing in NRI.
"""
import torch
import numpy as np
from itertools import permutations
from torch.utils.data import Dataset

def online_load_nri(data: dict, size: int):
    # edge list of a fully-connected graph
    es = np.array(list(permutations(range(size), 2))).T

    # convert the original data to torch.Tensor
    #data = {key: preprocess(value, es) for key, value in data.items()}

    # there is only a single key: 'train'
    adj = data['train'][0]
    adj = (adj + 1) / 2
    row, col = es
    # adjacency matrix in the form of edge list
    adj = adj[:, row, col]
    # organize the data in the batch form
    adj = torch.LongTensor(adj)
    state = torch.FloatTensor(data['train'][1])
    data = {'train': (adj, state)}

    #print("adj, state: ", adj.shape, state.shape) # [15000, 20], [15000, 60, 5, 4]
    #input("here")

    data, max_min = loc_vel(data)
    _, state = data['train']
    loc = state[:, :, :, :2]
    vel = state[:, :, :, 2:]
    loc_max = loc.max()
    loc_min = loc.min()
    vel_max = vel.max()
    vel_min = vel.min()
    for key, value in data.items():
        # normalize the location
        data[key][1][:, :, :, :2] = normalize(value[1][:, :, :, :2], loc_max, loc_min)
        # normalize the velocity
        data[key][1][:, :, :, 2:] = normalize(value[1][:, :, :, 2:], vel_max, vel_min)
    return data, es

def load_nri(data: dict, size: int):
    """
    Load Springs / Charged dataset.

    Args:
        data: train / val / test
        size: number of nodes, used for generating the edge list
    
    Return:
        data: min-max normalized data
        es: edge list
        max_min: maximum and minimum values of each input dimension
    """
    # edge list of a fully-connected graph
    es = np.array(list(permutations(range(size), 2))).T
    # convert the original data to torch.Tensor
    data = {key: preprocess(value, es) for key, value in data.items()}
    # for spring and charged
    # return maximum and minimum values of each input dimension in order to map normalized data back to the original space
    data, max_min = loc_vel(data)
    return data, es, max_min


def load_kuramoto(data: dict, size: int):
    """
    Load Kuramoto dataset.

    Args:
        data: train / val / test
        size: number of nodes, used for generating the edge list
    
    Return:
        data: min-max normalized data
        es: edge list
        max_min: maximum and minimum values of each input dimension
    """
    # edge list of a fully-connected graph
    es = np.array(list(permutations(range(size), 2))).T
    # convert the original data to torch.Tensor
    data = {key: preprocess(value, es) for key, value in data.items()}
    # return maximum and minimum values of each input dimension in order to map normalized data back to the original space
    data, max_min = dim_norm(data)
    # selected features, same as in NRI
    for key, value in data.items():
        data[key] = (value[0], value[1][:, :, :, [0, 1, 3]])
    max_min = [m[:, :, :, [0, 1, 3]] for m in max_min]
    return data, es, max_min


def dim_norm(data: dict):
    """
    Normalize node states in each dimension separately.

    Args:
        data: train / val / test, each in the form of [adj, state]

    Return:
        data: normalized data
        min_max: maximum and minimum values over all dimensions
    """
    # state: [batch, steps, node, dim]
    _, state = data['train']
    # maximum values over all dimensions
    M = max_except(state)
    # minimum values over all dimensions
    m = -max_except(-state)
    for key, value in data.items():
        try:
            data[key][1] = normalize(value[1], M, m)
        except:
            data[key] = (value[0], normalize(value[1], M, m))
    return data, (M, m)


def max_except(x: np.ndarray) -> np.ndarray:
    """
    Return the maximum values of x for each dimension over all samples.
    """
    shape = x.shape
    x = x.reshape((-1, shape[-1]))
    x, _ = x.max(0)
    size = [1] * len(shape[:-1]) + [shape[-1]]
    x = x.view(*size)
    return x


def loc_vel(data: dict):
    """
    Normalize Springs / Charged dataset (2-D dyamical systems). The dimension of the input feature is 4, 2 for location and 2 for velocity.

    Args:
        data: train / val / test

    Return:
        data: normalized data
        min_max: minimum and maximum values of each dimension of features in the training set
    """
    _, state = data['train']
    loc = state[:, :, :, :2]
    vel = state[:, :, :, 2:]
    loc_max = loc.max()
    loc_min = loc.min()
    vel_max = vel.max()
    vel_min = vel.min()
    for key, value in data.items():
        # normalize the location
        data[key][1][:, :, :, :2] = normalize(value[1][:, :, :, :2], loc_max, loc_min)
        # normalize the velocity
        data[key][1][:, :, :, 2:] = normalize(value[1][:, :, :, 2:], vel_max, vel_min)
    return data, (loc_max, loc_min, vel_max, vel_min)


def normalize(x: np.ndarray, up: float, down: float, a: float=-1, b: float=1) -> np.ndarray:
    """Scale the data x bounded in [down, up] to [a, b]."""
    return (x - down) / (up - down) * (b - a) + a


def preprocess(data: list, es: np.ndarray):
    """
    Convert the original data to torch.Tensor and organize them in the batch form.

    Args:
        data: [[adj, states], ...], all samples, each contains an adjacency matrix and the node states
        es: edge list

    Return:
        adj: adjacency matrices in the batch form
        states: node states in the batch form
    """
    # data: [[adj, states]]
    adj, state = [np.stack(i, axis=0) for i in zip(*data)]
    # scale the adjacency matrix to {0, 1}, only effective for Charged dataset since the elements take values in {-1, 1}
    adj = (adj + 1) / 2
    row, col = es
    # adjacency matrix in the form of edge list
    adj = adj[:, row, col]
    # organize the data in the batch form
    adj = torch.LongTensor(adj)
    state = torch.FloatTensor(state)

    #print("adj, state: ", adj.shape, state.shape) # [50000, 20], [num samples=50000, num timesteps=49, num nodes=5, num features=4]
    #input("here")
    return adj, state


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
