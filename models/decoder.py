from torch import Tensor, nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from models.gnn import GNN
from models.base import LinAct
from utils.metric import nll_gaussian
import config as cfg


class GRUX(nn.Module):
    """
    GRU from https://github.com/ethanfetaya/NRI/modules.py.
    """
    def __init__(self, dim_in: int, dim_hid: int, bias: bool=True):
        """
        Args:
            dim_in: input dimension
            dim_hid: dimension of hidden layers
            bias: adding a bias term or not, default: True
        """
        super(GRUX, self).__init__()
        self.hidden = nn.ModuleList([
            nn.Linear(dim_hid, dim_hid, bias)
            for _ in range(3)
        ])
        self.input = nn.ModuleList([
            nn.Linear(dim_in, dim_hid, bias)
            for _ in range(3)
        ])

    def forward(self, inputs: Tensor, hidden: Tensor, state: Tensor=None) -> Tensor:
        """
        Args:
            inputs: [..., dim]
            hidden: [..., dim]
            state: [..., dim], default: None
        """
        r = torch.sigmoid(self.input[0](inputs) + self.hidden[0](hidden))
        i = torch.sigmoid(self.input[1](inputs) + self.hidden[1](hidden))
        n = torch.tanh(self.input[2](inputs) + r * self.hidden[2](hidden))
        if state is None:
            state = hidden
        output = (1 - i) * n + i * state
        return output


class GNNDEC(GNN):
    """
    MLPDecoder of NRI from https://github.com/ethanfetaya/NRI/modules.py.
    """
    def __init__(self, n_in_node: int, edge_types: int,
                 msg_hid: int, msg_out: int, n_hid: int,
                 do_prob: float=0., skip_first: bool=False):
        """
        Args:
            n_in_node: input dimension
            edge_types: number of edge types
            msg_hid, msg_out, n_hid: dimension of different hidden layers
            do_prob: rate of dropout, default: 0
            skip_first: setting the first type of edge as non-edge or not, if yes, the first type of edge will have no effect, default: False
        """
        super(GNNDEC, self).__init__()
        self.msgs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_in_node, msg_hid),
                nn.ReLU(),
                nn.Dropout(do_prob),
                nn.Linear(msg_hid, msg_out),
                nn.ReLU(),
            )
            for _ in range(edge_types)
        ])

        self.out = nn.Sequential(
            nn.Linear(n_in_node + msg_out, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_in_node)
        )

        self.msg_out = msg_out
        self.skip_first = skip_first

    def move(self, x: Tensor, es: Tensor, z: Tensor) -> Tensor:
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
            z: [E, batch, K]

        Return:
            x: [node, batch, step, dim], future node states
        """
        # z: [E, batch, K] -> [E, batch, step, K]
        z = z.repeat(x.size(2), 1, 1, 1).permute(1, 2, 0, 3).contiguous()
        msg, col, size = self.message(x, es)
        idx = 1 if self.skip_first else 0
        norm = len(self.msgs)
        if self.skip_first:
            norm -= 1
        msgs = sum(self.msgs[i](msg) * torch.select(z, -1, i).unsqueeze(-1) / norm
                   for i in range(idx, len(self.msgs)))
        # aggregate all msgs from the incoming edges
        msgs = self.aggregate(msgs, col, size, 'add')
        # skip connection
        h = torch.cat([x, msgs], dim=-1)
        # predict the change in states
        delta = self.out(h)
        return x + delta

    def forward(self, x: Tensor, z: Tensor, es: Tensor, M: int=1) -> Tensor:
        """
        Args:
            x: [batch, step, node, dim], historical node states
            z: [E, batch, K], distribution of edge types
            es: [2, E], edge list
            M: number of steps to predict

        Return:
            future node states
        """
        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        x = x.permute(2, 0, 1, -1).contiguous()
        assert (M <= x.shape[2])
        # only take m-th timesteps as starting points (m: pred_steps)
        #x_m = x[:, :, 0::M, :] # default

        # NOTE: lines below are for online learning
        x_m = x[:, :, 0::(M-1), :] # M: 30 (default), taking t=0, t=29, t=58 data and then remove t=58 data
        x_m = x_m[:,:,:-1,:] # so that prediction starts at t=0 and t=29

        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        # predict M steps.
        xs = []
        for _ in range(0, M):
            x_m = self.move(x_m, es, z)
            xs.append(x_m)

        node, batch, skip, dim = xs[0].shape
        sizes = [node, batch, skip * M, dim]
        x_hat = Variable(torch.zeros(sizes))
        if x.is_cuda:
            x_hat = x_hat.cuda()
        # re-assemble correct timeline
        for i in range(M):
            #x_hat[:, :, i::M, :] = xs[i] # default

            # NOTE: lines below are for online learning
            if i < M-1:
                x_hat[:, :, i, :] = xs[i][:,:,0] # prediction at t=0 -> t=1, 2, ... , 29
                x_hat[:, :, i+M-1, :] = xs[i][:,:,1] # prediction at t=29 -> t=30, 31, ... , 58
            else:
                x_hat[:, :, i+M-1, :] = xs[i][:,:,1] # prediction at t=29 -> t=59

        x_hat = x_hat.permute(1, 2, 0, -1).contiguous()
        return x_hat[:, :(x.size(2) - 1)]


class RNNDEC(GNN):
    """
    RNN decoder with spatio-temporal message passing mechanisms.
    """
    def __init__(self, n_in_node: int, edge_types: int,
                 msg_hid: int, msg_out: int, n_hid: int,
                 do_prob: float=0., skip_first: bool=False, option='both'):
        """
        Args:
            n_in_node: input dimension
            edge_types: number of edge types
            msg_hid, msg_out, n_hid: dimension of different hidden layers
            do_prob: rate of dropout, default: 0
            skip_first: setting the first type of edge as non-edge or not, if yes, the first type of edge will have no effect, default: False
            option: default: 'both'
                'both': using both node-level and edge-level spatio-temporal message passing operations
                'node': using node-level the spatio-temporal message passing operation
                'edge': using edge-level the spatio-temporal message passing operation
        """
        super(RNNDEC, self).__init__()
        self.option = option
        self.msgs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_in_node, msg_hid),
                nn.ReLU(),
                nn.Dropout(do_prob),
                nn.Linear(msg_hid, msg_out),
                nn.ReLU(),
            )
            for _ in range(edge_types)
        ])

        self.out = nn.Sequential(
            nn.Linear(n_in_node + msg_out, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_in_node)
        )
        self.gru_edge = GRUX(n_hid, n_hid)
        self.gru_node = GRUX(n_hid + n_in_node, n_hid + n_in_node)
        self.msg_out = msg_out
        self.skip_first = skip_first
        print('Using learned interaction net decoder.')

    def move(self, x: Tensor, es: Tensor, z: Tensor, h_node: Tensor=None, h_edge: Tensor=None):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
            z: [E, batch, K]
            h_node: [node, batch, step, dim], hidden states of nodes, default: None
            h_edge: [E, batch, step, dim], hidden states of edges, default: None

        Return:
            x: [node, batch, step, dim], future node states
            msgs: [E, batch, step, dim], hidden states of edges
            cat: [node, batch, step, dim], hidden states of nodes
        """

        # z: [E, batch, K] -> [E, batch, step, K]
        z = z.repeat(x.size(2), 1, 1, 1).permute(1, 2, 0, 3).contiguous()
        msg, col, size = self.message(x, es)
        idx = 1 if self.skip_first else 0
        norm = len(self.msgs)

        if self.skip_first:
            norm -= 1
        msgs = sum(self.msgs[i](msg) * torch.select(z, -1, i).unsqueeze(-1) / norm
                   for i in range(idx, len(self.msgs)))
        if h_edge is not None and self.option in {'edge', 'both'}:
            msgs = self.gru_edge(msgs, h_edge)
        # aggregate all msgs from the incoming edges
        msg = self.aggregate(msgs, col, size)
        # skip connection
        cat = torch.cat([x, msg], dim=-1)
        if h_node is not None and self.option in {'node', 'both'}:
            cat = self.gru_node(cat, h_node)
        delta = self.out(cat)
        if self.option == 'node':
            msgs = None
        if self.option == 'edge':
            cat = None
        return x + delta, cat, msgs

    def forward(self, x: Tensor, z: Tensor, es: Tensor, M: int=1) -> Tensor:
        """
        Args:
            x: [batch, step, node, dim], historical node states
            z: [E, batch, K], distribution of edge types
            es: [2, E], edge list
            M: number of steps to predict

        Return:
            future node states
        """

        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        x = x.permute(2, 0, 1, -1).contiguous()
        assert (M <= x.shape[2])
        # only take m-th timesteps as starting points (m: pred_steps)
        #x_m = x[:, :, 0::M, :]

        # NOTE: lines below are for online learning
        x_m = x[:, :, 0::(M-1), :] # M: 30 (default), taking t=0, t=29, t=58 data and then remove t=58 data
        x_m = x_m[:,:,:-1,:] # so that prediction starts at t=0 and t=29

        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        # predict m steps.
        xs = []
        h_node, h_edge = None, None
        for _ in range(0, M):
            x_m, h_node, h_edge = self.move(x_m, es, z, h_node, h_edge)
            xs.append(x_m)

        node, batch, skip, dim = xs[0].shape
        sizes = [node, batch, skip * M, dim]
        x_hat = Variable(torch.zeros(sizes))
        if x.is_cuda:
            x_hat = x_hat.cuda()

        # Re-assemble correct timeline
        for i in range(M):
            #x_hat[:, :, i::M, :] = xs[i] # default

            # NOTE: lines below are for online learning
            if i < M-1:
                x_hat[:, :, i, :] = xs[i][:,:,0] # prediction at t=0 -> t=1, 2, ... , 29
                x_hat[:, :, i+M-1, :] = xs[i][:,:,1] # prediction at t=29 -> t=30, 31, ... , 58
            else:
                x_hat[:, :, i+M-1, :] = xs[i][:,:,1] # prediction at t=29 -> t=59

        x_hat = x_hat.permute(1, 2, 0, -1).contiguous()
        return x_hat[:, :(x.size(2) - 1)]


class OnlineRNNDEC(GNN):
    """
    RNN decoder with spatio-temporal message passing mechanisms.
    """
    def __init__(self, n_in_node: int, edge_types: int,
                 msg_hid: int, msg_out: int, n_hid: int,
                 do_prob: float=0., skip_first: bool=False, option='both'):
        """
        Args:
            n_in_node: input dimension
            edge_types: number of edge types
            msg_hid, msg_out, n_hid: dimension of different hidden layers
            do_prob: rate of dropout, default: 0
            skip_first: setting the first type of edge as non-edge or not, if yes, the first type of edge will have no effect, default: False
            option: default: 'both'
                'both': using both node-level and edge-level spatio-temporal message passing operations
                'node': using node-level the spatio-temporal message passing operation
                'edge': using edge-level the spatio-temporal message passing operation
        """
        super(OnlineRNNDEC, self).__init__()
        self.option = option
        self.msgs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_in_node, msg_hid),
                nn.ReLU(),
                nn.Dropout(do_prob),
                nn.Linear(msg_hid, msg_out),
                nn.ReLU(),
            )
            for _ in range(edge_types)
        ])

        self.out = nn.Sequential(
            nn.Linear(n_in_node + msg_out, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_in_node)
        )
        self.gru_edge = GRUX(n_hid, n_hid)
        self.gru_node = GRUX(n_hid + n_in_node, n_hid + n_in_node)
        self.msg_out = msg_out
        self.skip_first = skip_first
        #self.edges = nn.Parameter(torch.ones((15*14,1,2)) / 2).cuda()

        print('Using learned interaction net decoder.')

    def move(self, x: Tensor, es: Tensor, z: Tensor, h_node: Tensor=None, h_edge: Tensor=None):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
            z: [E, batch, K]
            h_node: [node, batch, step, dim], hidden states of nodes, default: None
            h_edge: [E, batch, step, dim], hidden states of edges, default: None

        Return:
            x: [node, batch, step, dim], future node states
            msgs: [E, batch, step, dim], hidden states of edges
            cat: [node, batch, step, dim], hidden states of nodes
        """
        # z: [E, batch, K] -> [E, batch, step, K]
        z = z.repeat(x.size(2), 1, 1, 1).permute(1, 2, 0, 3).contiguous()
        #print("[decoder] z: ", z) # probabilities
        #input("here")
        msg, col, size = self.message(x, es)
        idx = 1 if self.skip_first else 0
        norm = len(self.msgs)

        if self.skip_first:
            norm -= 1

        ##### multiple edge prediciton #####
        if x.shape[1] == 1:
            msg = msg.repeat(1,z.shape[1],1,1)
            x = x.repeat(1,z.shape[1],1,1)

        msgs = sum(self.msgs[i](msg) * torch.select(z, -1, i).unsqueeze(-1) / norm
                   for i in range(idx, len(self.msgs)))

        if h_edge is not None and self.option in {'edge', 'both'}:
            msgs = self.gru_edge(msgs, h_edge)

        # aggregate all msgs from the incoming edges
        msg = self.aggregate(msgs, col, size)
        # skip connection
        cat = torch.cat([x, msg], dim=-1)
        if h_node is not None and self.option in {'node', 'both'}:
            cat = self.gru_node(cat, h_node)
        delta = self.out(cat)

        if self.option == 'node':
            msgs = None
        if self.option == 'edge':
            cat = None

        return x + delta, cat, msgs

    def old_forward(self, x: Tensor, z: Tensor, es: Tensor, M: int=1, target=None, update_z=False) -> Tensor:
        """
        Args:
            x: [batch, step, node, dim], historical node states
            z: [E, batch, K], distribution of edge types
            es: [2, E], edge list
            M: number of steps to predict

        Return:
            future node states
        """

        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        x = x.permute(2, 0, 1, -1).contiguous()
        assert (M <= x.shape[2])
        # only take m-th timesteps as starting points (m: pred_steps)
        x_m = x[:, :, 0::M, :]

        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        # predict m steps.
        xs = []
        h_node, h_edge = None, None
        for i in range(0, M):
            x_m, h_node, h_edge = self.move(x_m, es, z, h_node, h_edge)
            xs.append(x_m)
            
            if (target is not None) and (update_z == True):
                lr_m = 2e+4
                target_m = target[:,i::M,:,:].permute(2,0,1,3)

                ##### multiple edge prediciton #####
                target_m = target_m.repeat(1,x_m.shape[1],1,1)

                #print("x_m, target_m: ", x_m.shape, target_m.shape)
                #input("here")

                relative_target_m = target_m.repeat(1,target_m.shape[0],1,1) - target_m.permute(1,0,2,3).repeat(target_m.shape[0],1,1,1)
                relative_x_m = x_m.repeat(1,x_m.shape[0],1,1) - x_m.permute(1,0,2,3).repeat(x_m.shape[0],1,1,1)

                # gradient on the sum of loss
                if i == (M-1):
                    derivative_m = torch.autograd.grad(F.mse_loss(x_m[:,:,:-1,:], target_m), z, retain_graph=True)
                    #derivative_m = torch.autograd.grad(F.l1_loss(x_m[:,:,:-1,:], target_m), z, retain_graph=True)
                    #derivative_m = torch.autograd.grad(F.mse_loss(x_m[:,:,:-1,:], target_m) + F.mse_loss(relative_x_m[:,:,:-1,:], relative_target_m), z, retain_graph=True)
                    #derivative_m = torch.autograd.grad(nll_gaussian(target_m.permute(2,1,0,3), x_m[:,:,:-1,:].permute(2,1,0,3), 5e-5), z, retain_graph=True)
                else:
                    derivative_m = torch.autograd.grad(F.mse_loss(x_m, target_m), z, retain_graph=True)
                    #derivative_m = torch.autograd.grad(F.l1_loss(x_m, target_m), z, retain_graph=True)
                    #derivative_m = torch.autograd.grad(F.mse_loss(x_m, target_m) + F.mse_loss(relative_x_m, relative_target_m), z, retain_graph=True)
                    #derivative_m = torch.autograd.grad(nll_gaussian(target_m.permute(2,1,0,3), x_m.permute(2,1,0,3), 5e-5), z, retain_graph=True)

                #z = z - (derivative_m[0] * 0.5 + derivative_m2[0] * 0.5) * lr_m
                z = z - derivative_m[0] * lr_m 

                z[z < 0] = 0
                z[z > 1] = 1

        node, batch, skip, dim = xs[0].shape
        sizes = [node, batch, skip * M, dim]
        x_hat = Variable(torch.zeros(sizes))
        if x.is_cuda:
            x_hat = x_hat.cuda()
        # Re-assemble correct timeline
        for i in range(M):
            x_hat[:, :, i::M, :] = xs[i]

        x_hat = x_hat.permute(1, 2, 0, -1).contiguous()

        if (target is not None) and (update_z == True):
            return x_hat[:, :(x.size(2) - 1)], z
        else:
            return x_hat[:, :(x.size(2) - 1)], None

    def forward(self, x: Tensor, z: Tensor, es: Tensor, M: int=1, target=None, update_z=False) -> Tensor:
        """
        Args:
            x: [batch, step, node, dim], historical node states
            z: [E, batch, K], distribution of edge types
            es: [2, E], edge list
            M: number of steps to predict

        Return:
            future node states
        """

        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        x = x.permute(2, 0, 1, -1).contiguous()
        assert (M <= x.shape[2])

        total_steps = x.shape[2]
        h_node, h_edge = None, None
        x_step_list = []

        for i in range(0, total_steps):
            if i < (total_steps - M):
                x_step, h_node, h_edge = self.move(x[:,:,i,:].unsqueeze(2), es, z, h_node, h_edge)
            else:
                x_step, h_node, h_edge = self.move(x_step_list[-1], es, z, h_node, h_edge)
            x_step_list.append(x_step)
        
        x_hat = torch.cat(x_step_list, dim=2)
        x_hat = x_hat.permute(1, 2, 0, -1).contiguous()

        return x_hat[:, :(x.size(2) - 1)]

class OnlineRNNDECV2(GNN):
    """
    RNN decoder with spatio-temporal message passing mechanisms.
    """
    def __init__(self, n_in_node: int, edge_types: int,
                 msg_hid: int, msg_out: int, n_hid: int,
                 do_prob: float=0., skip_first: bool=False, option='both'):
        """
        Args:
            n_in_node: input dimension
            edge_types: number of edge types
            msg_hid, msg_out, n_hid: dimension of different hidden layers
            do_prob: rate of dropout, default: 0
            skip_first: setting the first type of edge as non-edge or not, if yes, the first type of edge will have no effect, default: False
            option: default: 'both'
                'both': using both node-level and edge-level spatio-temporal message passing operations
                'node': using node-level the spatio-temporal message passing operation
                'edge': using edge-level the spatio-temporal message passing operation
        """
        super(OnlineRNNDECV2, self).__init__()
        self.option = option
        self.msgs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_in_node, msg_hid),
                nn.ReLU(),
                nn.Dropout(do_prob),
                nn.Linear(msg_hid, msg_out),
                nn.ReLU(),
            )
            for _ in range(edge_types)
        ])

        self.out = nn.Sequential(
            nn.Linear(n_in_node + msg_out, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_in_node)
        )
        self.gru_edge = GRUX(n_hid, n_hid)
        self.gru_node = GRUX(n_hid + n_in_node, n_hid + n_in_node)
        self.msg_out = msg_out
        self.skip_first = skip_first
        print('Using learned interaction net decoder.')

        self.fc = torch.randn((512, 64), requires_grad=True)
        self.fc2 = torch.randn((64, 1), requires_grad=True)
        self.A = 0.1 * torch.randn((256,1), requires_grad=True)
        self.B = 0.1 * torch.randn((256,1), requires_grad=True)
        self.C = 0.1 * torch.randn((512,1), requires_grad=True)
        self.D = 0.1 * torch.randn((1), requires_grad=True)

    def move(self, x: Tensor, es: Tensor, z: Tensor, h_node: Tensor=None, h_edge: Tensor=None):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
            z: [E, batch, K]
            h_node: [node, batch, step, dim], hidden states of nodes, default: None
            h_edge: [E, batch, step, dim], hidden states of edges, default: None

        Return:
            x: [node, batch, step, dim], future node states
            msgs: [E, batch, step, dim], hidden states of edges
            cat: [node, batch, step, dim], hidden states of nodes
        """
        # z: [E, batch, K] -> [E, batch, step, K]
        z = z.repeat(x.size(2), 1, 1, 1).permute(1, 2, 0, 3).contiguous()
        #print("[decoder] z: ", z) # probabilities
        #input("here")
        msg, col, size = self.message(x, es)
        idx = 1 if self.skip_first else 0
        norm = len(self.msgs)

        if self.skip_first:
            norm -= 1

        msgs_before = self.msgs[1](msg)
        msgs = sum(self.msgs[i](msg) * torch.select(z, -1, i).unsqueeze(-1) / norm
                   for i in range(idx, len(self.msgs)))
        msgs_after = msgs

        if h_edge is not None and self.option in {'edge', 'both'}:
            msgs = self.gru_edge(msgs, h_edge)
        # aggregate all msgs from the incoming edges
        msg = self.aggregate(msgs, col, size)
        # skip connection
        cat = torch.cat([x, msg], dim=-1)
        if h_node is not None and self.option in {'node', 'both'}:
            cat = self.gru_node(cat, h_node)
        delta = self.out(cat)
        if self.option == 'node':
            msgs = None
        if self.option == 'edge':
            cat = None
        return x + delta, cat, msgs, (msgs_before, msgs_after)

    def forward(self, x: Tensor, z: Tensor, es: Tensor, M: int=1, target=None, encode=False) -> Tensor:
        """
        Args:
            x: [batch, step, node, dim], historical node states
            z: [E, batch, K], distribution of edge types
            es: [2, E], edge list
            M: number of steps to predict

        Return:
            future node states
        """

        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        x = x.permute(2, 0, 1, -1).contiguous()
        assert (M <= x.shape[2])
        # only take m-th timesteps as starting points (m: pred_steps)
        x_m = x[:, :, 0::M, :]

        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        # predict m steps.
        xs = []
        h_node, h_edge = None, None
        for i in range(0, M):
            x_m, h_node, h_edge, msgs_bf_af = self.move(x_m, es, z, h_node, h_edge)
            xs.append(x_m)

            if (target is not None) and (encode == True):
                # if str(self.A.device) == 'cpu':
                #     self.A = self.A.to(z.device)
                #     self.B = self.B.to(z.device)
                #     self.C = self.C.to(z.device)
                #     self.D = self.D.to(z.device)
                if str(self.fc.device) == 'cpu':
                    self.fc = self.fc.to(z.device)
                    self.fc2 = self.fc2.to(z.device)
 
                if i > 0:
                    target_m = target[:,i::M,:,:].permute(2,0,1,3)

                    if i == (M-1):
                        # derivative_A = torch.autograd.grad(F.mse_loss(x_m[:,:,:-1,:], target_m), self.A, retain_graph=True) 
                        # derivative_B = torch.autograd.grad(F.mse_loss(x_m[:,:,:-1,:], target_m), self.B, retain_graph=True) 
                        # derivative_C = torch.autograd.grad(F.mse_loss(x_m[:,:,:-1,:], target_m), self.C, retain_graph=True) 
                        # derivative_D = torch.autograd.grad(F.mse_loss(x_m[:,:,:-1,:], target_m), self.D, retain_graph=True) 
                        derivative_fc = torch.autograd.grad(F.mse_loss(x_m[:,:,:-1,:], target_m), self.fc, retain_graph=True) 
                        derivative_fc2 = torch.autograd.grad(F.mse_loss(x_m[:,:,:-1,:], target_m), self.fc2, retain_graph=True) 
                    else:
                        # derivative_A = torch.autograd.grad(F.mse_loss(x_m, target_m), self.A, retain_graph=True)
                        # derivative_B = torch.autograd.grad(F.mse_loss(x_m, target_m), self.B, retain_graph=True)
                        # derivative_C = torch.autograd.grad(F.mse_loss(x_m, target_m), self.C, retain_graph=True)
                        # derivative_D = torch.autograd.grad(F.mse_loss(x_m, target_m), self.D, retain_graph=True)
                        derivative_fc = torch.autograd.grad(F.mse_loss(x_m, target_m), self.fc, retain_graph=True)
                        derivative_fc2 = torch.autograd.grad(F.mse_loss(x_m, target_m), self.fc2, retain_graph=True)

                    lr = 7.5
                    # self.A = self.A - lr * derivative_A[0]
                    # self.B = self.B - lr * derivative_B[0]
                    # self.C = self.C - lr * derivative_C[0]
                    # self.D = self.D - lr * derivative_D[0]
                    self.fc = self.fc - lr * derivative_fc[0]
                    self.fc2 = self.fc2 - lr * derivative_fc2[0]

                # update = F.tanh(torch.matmul(msgs_bf_af[0], self.A) \
                #             + torch.matmul(msgs_bf_af[1], self.B) \
                #             + torch.matmul(torch.cat((msgs_bf_af[0], msgs_bf_af[1]), dim=-1), self.C) \
                #             + self.D)
                # update = update.sum(dim=(2,3)) / update.shape[2]              

                update = F.relu(torch.matmul(torch.cat((msgs_bf_af[0], msgs_bf_af[1]), dim=-1), self.fc))
                update = F.tanh(torch.matmul(update, self.fc2))
                update = update.sum(dim=(2,3)) / update.shape[2]              

                lr = 0.1
                z[:,:,1] = z[:,:,1] + lr * update
                z[z < 0] = 0
                z[z > 1] = 1


        node, batch, skip, dim = xs[0].shape
        sizes = [node, batch, skip * M, dim]
        x_hat = Variable(torch.zeros(sizes))
        if x.is_cuda:
            x_hat = x_hat.cuda()
        # Re-assemble correct timeline
        for i in range(M):
            x_hat[:, :, i::M, :] = xs[i]

        x_hat = x_hat.permute(1, 2, 0, -1).contiguous()

        if (target is not None) and (encode == True):
            return x_hat[:, :(x.size(2) - 1)], z
        else:
            return x_hat[:, :(x.size(2) - 1)], None

class AttDEC(GNN):
    """
    Spatio-temporal message passing mechanisms implemented by combining RNNs and attention mechanims.
    """
    def __init__(self, n_in_node: int, edge_types: int,
                 msg_hid: int, msg_out: int, n_hid: int,
                 do_prob: float=0., skip_first: bool=False):
        """
        Args:
            n_in_node: input dimension
            edge_types: number of edge types
            msg_hid, msg_out, n_hid: dimension of different hidden layers
            do_prob: rate of dropout, default: 0
            skip_first: setting the first type of edge as non-edge or not, if yes, the first type of edge will have no effect, default: False
        """
        super(AttDEC, self).__init__()
        self.input_emb = nn.Linear(n_in_node, cfg.input_emb_hid)
        self.msgs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * (n_in_node + cfg.input_emb_hid), msg_hid),
                nn.ReLU(),
                nn.Dropout(do_prob),
                nn.Linear(msg_hid, msg_out),
                nn.ReLU(),
            )
            for _ in range(edge_types)
        ])

        self.out = nn.Sequential(
            nn.Linear(n_in_node + msg_out, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_in_node)
        )
        self.gru_edge = GRUX(msg_out, msg_out)
        self.gru_node = GRUX(n_in_node + msg_out, n_in_node + msg_out)
        self.msg_out = msg_out
        self.skip_first = skip_first

        # attention mechanism
        self.attn = nn.Linear(msg_out + n_in_node, cfg.att_hid)
        self.query = LinAct(n_in_node + msg_out, cfg.att_hid)
        self.key = LinAct(n_in_node + msg_out, cfg.att_hid)
        self.value = LinAct(n_in_node + msg_out, cfg.att_hid)
        self.att_out = LinAct(cfg.att_hid, n_in_node + msg_out)

    def temporalAttention(self, x: Tensor, h: Tensor):
        """
        Update hidden states of nodes by the temporal attention mechanism.

        Args:
            x: [step_att, node, batch, step, dim], historical hidden states of nodes used for temporal attention
            h: [node, batch, step, dim], hidden states of nodes from RNNs
        
        Return:
            output: [node, batch, step, dim], hidden states of nodes updated by the attention mechanism
            out_att: [batch, node, step, step_att], attentions of historical steps w.r.t. current step
        """
        # concatenate current hidden states of nodes with historical hidden states
        h_current = h.unsqueeze(0).contiguous()
        x = torch.cat([x, h_current], dim=0)

        # x: [step_att, node, batch, step, dim] -> [node, batch, step, step_att, dim]
        x = x.permute(1, 2, 3, 0, 4).contiguous()
        # [node, batch, step, 1, att_hid]
        query = self.query(h.unsqueeze(3))
        # [node, batch, step, step_att, att_hid]
        key = self.key(x)
        value = self.value(x)
        # key: [node, batch, step, step_att, att_hid] -> [node, batch, step, att_hid, step_att]
        key = key.transpose(-1, -2).contiguous()

        # [node, batch, step, 1, step_att]
        attention = torch.matmul(query, key) / (cfg.att_hid ** 0.5)
        attention = attention.softmax(-1)
        # [node, batch, step, att_hid]
        att_value = torch.matmul(attention, value).squeeze(3)
        output = self.att_out(att_value)

        # [batch, node, step, step_att]
        out_att = attention.squeeze(3).transpose(0, 1).contiguous()  
        return output, out_att

    def move(self, x, es, z, h_att, h_node=None, h_edge=None):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
            z: [E, batch, K]
            h_att: [step_att, node, batch, step, dim], historical hidden states of nodes used for temporal attention
            h_node: [node, batch, step, dim], hidden states of nodes, default: None
            h_edge: [E, batch, step, dim], hidden states of edges, default: None

        Return:
            x: [node, batch, step, dim], future node states
            h_att: [step_att + 1, node, batch, step, dim], accumulated historical hidden states of nodes used for temporal attention
            msgs: [E, batch, step, dim], hidden states of edges
            cat: [node, batch, step, dim], hidden states of nodes
        """
        x_emb = self.input_emb(x)
        x_emb = torch.cat([x_emb, x], dim=-1)
        # z: [E, batch, K] -> [E, batch, step, K]
        z = z.repeat(x.size(2), 1, 1, 1).permute(1, 2, 0, 3).contiguous()
        msg, col, size = self.message(x_emb, es)
        idx = 1 if self.skip_first else 0
        norm = len(self.msgs)
        if self.skip_first:
            norm -= 1
        msgs = sum(self.msgs[i](msg) * torch.select(z, -1, i).unsqueeze(-1) / norm
                   for i in range(idx, len(self.msgs)))

        if h_edge is not None:
            msgs = self.gru_edge(msgs, h_edge)
        # aggregate all msgs to receiver
        msg = self.aggregate(msgs, col, size)

        cat = torch.cat([x, msg], dim=-1)
        if h_node is None:
            delta = self.out(cat)
            h_att = cat.unsqueeze(0)
        else:
            cat = self.gru_node(cat, h_node)
            cur_hidden, _ = self.temporalAttention(h_att, cat)
            h_att = torch.cat([h_att, cur_hidden.unsqueeze(0)], dim=0)
            delta = self.out(cur_hidden)

        return x + delta, h_att, cat, msgs

    def forward(self, x: Tensor, z: Tensor, es: Tensor, M: int=1) -> Tensor:
        """
        Args:
            x: [batch, step, node, dim], historical node states
            z: [E, batch, K], distribution of edge types
            es: [2, E], edge list
            M: number of steps to predict

        Return:
            future node states
        """
        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        x = x.permute(2, 0, 1, -1).contiguous()
        assert (M <= x.shape[2])
        # only take m-th timesteps as starting points (m: pred_steps)
        #x_m = x[:, :, 0::M, :]

        # NOTE: lines below are for online learning
        x_m = x[:, :, 0::(M-1), :] # M: 30 (default), taking t=0, t=29, t=58 data and then remove t=58 data
        x_m = x_m[:,:,:-1,:] # so that prediction starts at t=0 and t=29

        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        # predict M steps.
        xs = []
        att_hidden, edge_hidden, node_hidden = None, None, None

        for _ in range(0, M):
            x_m, att_hidden, node_hidden, edge_hidden = self.move(x_m, es, z, att_hidden, node_hidden, edge_hidden)
            xs.append(x_m)

        node, batch, skip, dim = xs[0].shape
        sizes = [node, batch, skip * M, dim]
        x_hat = Variable(torch.zeros(sizes))

        if x.is_cuda:
            x_hat = x_hat.cuda()
        # re-assemble correct timeline
        for i in range(M):
            #x_hat[:, :, i::M, :] = xs[i] # default

            # NOTE: lines below are for online learning
            if i < M-1:
                x_hat[:, :, i, :] = xs[i][:,:,0] # prediction at t=0 -> t=1, 2, ... , 29
                x_hat[:, :, i+M-1, :] = xs[i][:,:,1] # prediction at t=29 -> t=30, 31, ... , 58
            else:
                x_hat[:, :, i+M-1, :] = xs[i][:,:,1] # prediction at t=29 -> t=59
        x_hat = x_hat.permute(1, 2, 0, -1).contiguous()
        return x_hat[:, :(x.size(2) - 1)]