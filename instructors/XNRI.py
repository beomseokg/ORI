import os
import torch
import torch.nn.functional as F
import config as cfg
import config_online as cfg_online
import torch.nn as nn
from torch.nn.functional import mse_loss
from utils.torch_extension import edge_accuracy, asym_rate, transpose, edge_f1, plot_motions, plot_motions_edge_static
from instructors.base import Instructor
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch import Tensor, optim
from utils.metric import cross_entropy, kl_divergence, nll_gaussian
from torch.optim.lr_scheduler import StepLR


import matplotlib.pyplot as plt
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

    x_max = max(target[:,:,0].max(), preds[:,:,0].max()).detach().cpu()
    x_min = max(target[:,:,0].min(), preds[:,:,0].min()).detach().cpu()
    y_max = max(target[:,:,1].max(), preds[:,:,1].max()).detach().cpu()
    y_min = max(target[:,:,1].min(), preds[:,:,1].min()).detach().cpu()

    f, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].scatter(x=target[:,:29,0].detach().cpu(), y=target[:,:29,1].detach().cpu(), c=colors[:,:29], alpha=0.1)
    ax[0].scatter(x=preds[:,29:,0].detach().cpu(), y=preds[:,29:,1].detach().cpu(), c=colors[:,29:], alpha=1)
    ax[1].scatter(x=target[:,:29,0].detach().cpu(), y=target[:,:29,1].detach().cpu(), c=colors[:,:29], alpha=0.1)
    ax[1].scatter(x=target[:,29:,0].detach().cpu(), y=target[:,29:,1].detach().cpu(), c=colors[:,29:], alpha=1)
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)
    ax[1].set_xlim(x_min, x_max)
    ax[1].set_ylim(y_min, y_max)
    plt.show()
    plt.savefig(path)
    plt.close()


def vec2mat(vector, size):
    test = torch.zeros((size,size))
    for i in range(size):
        if i == 0:
            test[0,1:] = vector.reshape(size,size-1)[0,:]
        elif i == (size-1):
            test[size-1,:(size-1)] = vector.reshape(size,size-1)[size-1,:]
        else:
            test[i,0:i] = vector.reshape(size,size-1)[i,0:i]
            test[i,(i+1):] = vector.reshape(size,size-1)[i,i:]
    return test

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

class XNRIIns(Instructor):
    """
    Training and testing for the neural relational inference task.
    """
    def __init__(self, model: torch.nn.DataParallel, data: dict, es: np.ndarray, cmd):
        """
        Args:
            model: an auto-encoder
            data: train / val /test set
            es: edge list
            cmd: command line parameters
        """
        super(XNRIIns, self).__init__(cmd)
        self.model = model
        if cmd.dyn != "motion":
            self.data = {key: TensorDataset(value[0], value[1])
                        for key, value in data.items()}
        else:
            self.data = data
        self.es = torch.LongTensor(es)
        # number of nodes
        self.size = cmd.size
        self.batch_size = cmd.batch
        self.args = cmd
        if self.args.online == True:
            # optimizer
            self.opt = optim.Adam([
                {'params': self.model.module.enc.parameters(), 'lr': cfg_online.lr},
                {'params': self.model.module.dec.parameters()}], lr=cfg_online.lr)
            # learning rate scheduler, same as in NRI
            self.scheduler = StepLR(self.opt, step_size=cfg_online.lr_decay, gamma=cfg_online.gamma)
        else:
            # optimizer
            self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)
            # learning rate scheduler, same as in NRI
            self.scheduler = StepLR(self.opt, step_size=cfg.lr_decay, gamma=cfg.gamma)

        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.cmd.size, self.cmd.size]) - np.eye(self.cmd.size)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)

        self.log.info("data: " + self.cmd.data_path)
        self.log.info("\n[info] Model named parameters")
        for name, params in self.model.module.enc.named_parameters():
            self.log.info("Encoder name: " + name + ' ' + str(params.shape))
        enc_num_params = sum(p.numel() for p in self.model.module.enc.parameters() if p.requires_grad)
        for name, params in self.model.module.dec.named_parameters():
            self.log.info("Decoder name: " + name + ' ' + str(params.shape))
        dec_num_params = sum(p.numel() for p in self.model.module.dec.parameters() if p.requires_grad)
        self.log.info('[Info] Number of Trainable Parameters in Encoder: {}'.format(enc_num_params))
        self.log.info('[Info] Number of Trainable Parameters in Decoder: {}\n'.format(dec_num_params))


    def train(self):
        # use the loss as the metric for model selection, default: +\infty
        val_best = np.inf
        # path to save the current best model
        prefix = '/'.join(cfg.log.split('/')[:-1])
        name = '{}/best.pth'.format(prefix)

        #print("skipping train and val...")
        #self.cmd.epochs = 0
        for epoch in range(1, 1 + self.cmd.epochs):
            self.model.train()

            if self.cmd.online == True:
                data = self.load_data(self.data['train'], self.batch_size, shuffle=False)
            else:
                data = self.load_data(self.data['train'], self.batch_size)

            loss_a = 0.
            mse_a = 0.
            acc_a = 0.
            tp_a, tn_a, fp_a, fn_a = 0., 0., 0., 0.,
            N = 0.
            batch_idx = 0
            agent_wise_crr = torch.zeros((5,5)).cuda()
            for adj, states in data: # iterating batches

                if cfg.gpu:
                    states = states.cuda()
                    adj = adj.cuda()
                scale = len(states) / self.batch_size

                #print("states: ", states.shape) # [1, 60, 5, 4]
                #print("scale: ", scale) # 1
                #input("here")

                # N: number of samples, equal to the batch size with possible exception for the last batch
                N += scale
                if self.args.online == True:
                    temp_loss, temp_mse, temp_acc, temp_test_loss, temp_test_mse, temp_test_acc, temp_agent_wise_crr, temp_f1_list = self.online_train_nri_v3(states, adj, batch_idx)
                    loss_a += scale * temp_test_loss
                    mse_a += scale * temp_test_mse
                    acc_a += scale * temp_test_acc

                    tp_a += temp_f1_list[0]
                    tn_a += temp_f1_list[1]
                    fp_a += temp_f1_list[2]
                    fn_a += temp_f1_list[3]

                    #self.log.info('batch {:03d} loss {:.3e} mse {:.7e} acc {:.5e}'.format(batch_idx, temp_test_loss, temp_test_mse, temp_test_acc))
                    self.log.info('batch {:03d} loss {:.3e} mse {:.7e} acc {:.5e}'.format(batch_idx, temp_test_loss, temp_test_mse, temp_test_acc) + \
                                  ' crr: ' + str(float(temp_agent_wise_crr.sum(dim=-1)[0])) + ' ' + str(float(temp_agent_wise_crr.sum(dim=-1)[1])) + ' ' + str(float(temp_agent_wise_crr.sum(dim=-1)[2])) + ' ' + str(float(temp_agent_wise_crr.sum(dim=-1)[3])) + ' ' + str(float(temp_agent_wise_crr.sum(dim=-1)[4])) + ' ' + \
                                  ' tp tn fp fn: ' + str(float(temp_f1_list[0])) + ' ' + str(float(temp_f1_list[1])) + ' ' + str(float(temp_f1_list[2])) + ' ' + str(float(temp_f1_list[3])) )
                                  #' var: ' + str(float(temp_test_var[0])) + ' ' + str(float(temp_test_var[1])) + ' ' + str(float(temp_test_var[2])) + ' ' + str(float(temp_test_var[3])) + ' ' + str(float(temp_test_var[4])) )

                else:
                    loss_a += scale * self.train_nri(states)
                    #print("skipping train ...")

                batch_idx += 1
            loss_a /= N 
            mse_a /= N
            acc_a /= N
            self.log.info('epoch {:03d} loss {:.3e} mse {:.7e} acc {:.5e}'.format(epoch, loss_a, mse_a, acc_a))
            self.log.info('tp {:.3e} tn {:.3e} fp {:.3e} fn {:.3e}'.format(tp_a, tn_a, fp_a, fn_a))

            if self.args.online == True:
                torch.save(self.model.module.state_dict(), name)

            else:
                #print("skipping val ...")
                #"""
                losses = self.report('val', [cfg.M])
                val_cur = losses[0]
                if val_cur < val_best:
                    # update the current best model when approaching a lower loss
                    self.log.info('epoch {:03d} metric {:.3e}'.format(epoch, val_cur))
                    val_best = val_cur
                    torch.save(self.model.module.state_dict(), name)
                #"""

            # learning rate scheduling
            self.scheduler.step()

        if self.cmd.epochs > 0:
            self.model.module.load_state_dict(torch.load(name))
        
        if self.args.online == False:
            self.test('test', 20)
            #input("test done")

    def report(self, name: str, Ms: list) -> list:
        """
        Evaluate the loss.

        Args:
            name: 'train' / 'val' / 'test'
            Ms: [...], each element is a number of steps to predict
        
        Return:
            losses: [...], each element is an average loss
        """
        losses = []
        for M in Ms:
            loss, mse, acc, rate, ratio, sparse = self.evalate(self.data[name], M)
            losses.append(loss)
            self.log.info('{} M {:02d} mse {:.3e} acc {:.4f} _acc {:.4f} rate {:.4f} ratio {:.4f} sparse {:.4f}'.format(
                name, M, mse, acc, 1 - acc, rate, ratio, sparse))
        return losses

    def single_batch_train(self):
        # use the loss as the metric for model selection, default: +\infty
        val_best = np.inf
        # path to save the current best model
        prefix = '/'.join(cfg.log.split('/')[:-1])
        name = '{}/best.pth'.format(prefix)
        os.mkdir(prefix+"/plot/")
        os.mkdir(prefix+"/edge/")

        self.path = prefix

        #print("skipping train and val...")
        #self.cmd.epochs = 0
        self.model.train()
        # shuffle the data at each epoch
        if self.cmd.dyn == "motion":
            data = self.data
        else:
            if self.cmd.online == True:
                data = self.load_data(self.data['train'], self.batch_size, shuffle=False)
            else:
                data = self.load_data(self.data['train'], self.batch_size)

        loss_a = 0.
        mse_a = 0.
        acc_a = 0.
        tp_a, tn_a, fp_a, fn_a = 0., 0., 0., 0.,
        N = 0.

        for batch_idx, (adj, states) in enumerate(data):
            if self.cmd.dyn == "motion":
                # adj <- states
                # states <- adj
                states = adj
                adj = torch.zeros((1,930,2))

            if cfg.gpu:
                states = states.cuda()
                adj = adj.cuda()
            scale = len(states) / self.batch_size

            # N: number of samples, equal to the batch size with possible exception for the last batch
            N += scale

            if self.cmd.dyn == "motion":
                num_iters = 4
            elif self.cmd.dyn == "bball":
                num_iters = 4
            else:
                num_iters = 2999

            for i in range(num_iters):
                start_idx = i * cfg_online.prediction_steps
                end_idx = start_idx + cfg_online.timesteps
                temp_data = states[:,start_idx:end_idx,:,:]
                temp_data_aug = temp_data.clone()

                if self.cmd.trajectory_flip == True:
                    temp_data_aug = temp_data_aug.repeat(4,1,1,1)
                    #flip = int(torch.randint(low=0, high=4, size=(1,)))
                    for flip in range(4):
                        if flip == 0: # same
                            temp_data_aug[flip,:,:,(0,2)] = temp_data[:,:,:,(0,2)]
                            temp_data_aug[flip,:,:,(1,3)] = temp_data[:,:,:,(1,3)]
                        elif flip == 1: # flip x-axis
                            temp_data_aug[flip,:,:,(0,2)] = temp_data[:,:,:,(0,2)] * -1
                            temp_data_aug[flip,:,:,(1,3)] = temp_data[:,:,:,(1,3)]
                        elif flip == 2: # flip y-axis
                            temp_data_aug[flip,:,:,(0,2)] = temp_data[:,:,:,(0,2)]
                            temp_data_aug[flip,:,:,(1,3)] = temp_data[:,:,:,(1,3)] * -1
                        elif flip == 3: # flip x and y-axis
                            temp_data_aug[flip,:,:,(0,2)] = temp_data[:,:,:,(0,2)] * -1
                            temp_data_aug[flip,:,:,(1,3)] = temp_data[:,:,:,(1,3)] * -1

                if self.args.online == True:
                    loss, mse, mse_list, acc = self.online_train_nri_v4(temp_data_aug, adj, i + batch_idx * num_iters)
                    loss_a += scale * loss.item()
                    mse_a += scale * mse.item()
                    acc_a += scale * acc

                    if self.cmd.dyn == "motion":
                        self.log.info('batch {:03d} loss {:.3e} mse {:.7e} '.format(i + batch_idx * num_iters, loss.item(), mse.item()) + \
                                        'mse t1 {:.7e} t5 {:.7e} t10 {:.7e} d_edges {:.5e} edge_lr {:d} '.format(mse_list[0].item(), mse_list[1].item(), mse_list[2].item(), float(self.delta_edges), self.cmd.edge_lr))
                    else:
                        self.log.info('batch {:03d} loss {:.3e} mse {:.7e} '.format(i + batch_idx * num_iters, loss.item(), mse.item()) + \
                                        'mse t1 {:.7e} t10 {:.7e} t20 {:.7e} t30 {:.7e} acc {:.5e} d_edges {:.5e} '.format(mse_list[0].item(), mse_list[1].item(), mse_list[2].item(), mse_list[3].item(), acc, float(self.delta_edges)) + \
                                        'mse_true {:.7e} mse_false {:.7e} edge_lr: {:d}'.format(mse_list[4].item(), mse_list[5].item(), self.cmd.edge_lr))

                else:
                    loss_a += scale * self.train_nri(states)
                    #print("skipping train ...")

        loss_a /= N 
        mse_a /= N
        acc_a /= N
        self.log.info('epoch {:03d} loss {:.3e} mse {:.7e} acc {:.5e}'.format(1, loss_a, mse_a, acc_a))
        self.log.info('tp {:.3e} tn {:.3e} fp {:.3e} fn {:.3e}'.format(tp_a, tn_a, fp_a, fn_a))

        if self.args.online == True:
            torch.save(self.model.module.state_dict(), name)
        else:
            #print("skipping val ...")
            #"""
            losses = self.report('val', [cfg.M])
            val_cur = losses[0]
            if val_cur < val_best:
                # update the current best model when approaching a lower loss
                self.log.info('epoch {:03d} metric {:.3e}'.format(epoch, val_cur))
                val_best = val_cur
                torch.save(self.model.module.state_dict(), name)
            #"""

        # learning rate scheduling
        #self.scheduler.step()

        if self.cmd.epochs > 0:
            self.model.module.load_state_dict(torch.load(name))
        
        if self.args.online == False:
            self.test('test', 20)
            #input("test done")

    def report(self, name: str, Ms: list) -> list:
        """
        Evaluate the loss.

        Args:
            name: 'train' / 'val' / 'test'
            Ms: [...], each element is a number of steps to predict
        
        Return:
            losses: [...], each element is an average loss
        """
        losses = []
        for M in Ms:
            loss, mse, acc, rate, ratio, sparse = self.evalate(self.data[name], M)
            losses.append(loss)
            self.log.info('{} M {:02d} mse {:.3e} acc {:.4f} _acc {:.4f} rate {:.4f} ratio {:.4f} sparse {:.4f}'.format(
                name, M, mse, acc, 1 - acc, rate, ratio, sparse))
        return losses

    def online_train_nri(self, states: Tensor, adj: Tensor) -> Tensor:
        """
        NOTE: the model is first optimized in the observed period (i.e., states[:cfg_online.train_steps]) -> optimized loss
              then, predicts the remaining future period (i.e., states[cfg_online.train_steps:]) -> evaluated loss 

        Args:
            states: [batch, step, node, dim], all node states, including historical states and the states to predict
        """
        # compute the relation distribution (prob) and predict future node states (output)
        self.model.train()

        states_enc = states[:, :cfg_online.train_steps, :, :]
        states_dec = states[:, :cfg_online.train_steps, :, :]
        target = states_dec[:, 1:, :, :]

        output, prob = self.model(states_enc, states_dec, p=True, M=10, tosym=cfg_online.sym)
        prob = prob.transpose(0, 1).contiguous()

        # reconstruction loss and the KL-divergence
        loss_nll = nll_gaussian(target, output, 5e-5)
        loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
        loss = loss_nll + loss_kl
        # impose the soft symmetric contraint by adding a regularization term
        if self.cmd.reg > 0:
            # transpose the relation distribution
            prob_hat = transpose(prob, self.size)
            loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
            loss = loss + loss_sym * self.cmd.reg
        self.optimize(self.opt, loss * cfg_online.scale)

        # choice for the evaluation metric, adding the regularization term or not
        # when the penalty factor is large, it may be misleading to add the regularization term
        if self.cmd.no_reg:
            loss = loss_nll + loss_kl

        mse = mse_loss(output, target).data
        acc = edge_accuracy(prob, adj)

        print("[onlinetrainnri] output, target, prob: ", output.shape, target.shape, prob.shape)
        input("here")

        """
        NOTE: Test starts from here
        """
        self.model.eval()

        states_enc = states[:, :cfg_online.train_steps, :, :]
        states_dec = states[:, cfg_online.train_steps-1:cfg_online.train_steps+cfg_online.M, :, :]
        target = states_dec[:, 1:, :, :]

        output, prob = self.model(states_enc, states_dec, p=True, M=cfg_online.M, tosym=cfg_online.sym)
        prob = prob.transpose(0, 1).contiguous()

        # reconstruction loss and the KL-divergence
        """
        test_loss_nll = nll_gaussian(target, output, 5e-5)
        test_loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
        test_loss = test_loss_nll + test_loss_kl
        # impose the soft symmetric contraint by adding a regularization term
        if self.cmd.reg > 0:
            # transpose the relation distribution
            prob_hat = transpose(prob, self.size)
            test_loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
            test_loss = test_loss + test_loss_sym * self.cmd.reg

        # choice for the evaluation metric, adding the regularization term or not
        # when the penalty factor is large, it may be misleading to add the regularization term
        if self.cmd.no_reg:
            test_loss = test_loss_nll + test_loss_kl
        """

        test_mse = mse_loss(output, target).data
        test_acc = edge_accuracy(prob, adj)

        return loss, mse, acc, 0, test_mse, test_acc

    def online_train_nri_v2(self, states: Tensor, adj: Tensor) -> Tensor:
        """
        NOTE: the model with initializied interaction matric (W=[0.5]) is optimized in the observed period
        The interaction matrix is fed into the decoder to predict the futurue period.

        Args:
            states: [batch, step, node, dim], all node states, including historical states and the states to predict
        """
        # compute the relation distribution (prob) and predict future node states (output)
        self.model.train()

        states_enc = states[:, :cfg_online.train_steps, :, :]
        states_dec = states[:, :cfg_online.train_steps, :, :]
        target_enc = states_enc[:, 1:, :, :,]
        target = states_dec[:, 1:, :, :]

        if not self.es.is_cuda:
            self.es = self.es.cuda(states_enc.device)

        with torch.no_grad():
            self.model.module.dec.W.fill_(0.5)
        output = self.model.module.dec(states_enc, self.model.module.dec.W, self.es, M=1)

        #first_derivative = torch.autograd.grad(F.mse_loss(output, target_enc), W, retain_graph=True)
        #W = W - 1e5 * first_derivative[0]
        #W = F.softmax(W, dim=-1)

        # reconstruction loss and the KL-divergence
        loss_nll = nll_gaussian(target_enc, output, 5e-5)
        loss = loss_nll

        #self.optimize(self.opt, loss * cfg_online.scale)

        self.opt.zero_grad()
        # first_derivative = torch.autograd.grad(loss, self.model.module.dec.W, retain_graph=True)
        # self.model.module.dec.W -= 1e-2*first_derivative
        # #print("first_derivative: ", first_derivative)
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            #print("self.model.module.dec.W: ", self.model.module.dec.W.grad)
            #print("self.model.module.dec.W: ", self.model.module.dec.W.grad / torch.max(torch.abs(self.model.module.dec.W.grad) * 2))
            #print("before W: ", self.model.module.dec.W)
            self.model.module.dec.W -= self.model.module.dec.W.grad / torch.max(torch.abs(self.model.module.dec.W.grad) * 2)
            #self.model.module.dec.W = F.softmax(self.model.module.dec.W, dim=-1)
            #print("after W: ", self.model.module.dec.W)
            #input("here")

        mse = mse_loss(output, target_enc).data
        acc = edge_accuracy(self.model.module.dec.W, adj)

        """
        NOTE: Test starts from here
        """
        self.model.eval()

        states_enc = states[:, :cfg_online.train_steps, :, :]
        states_dec = states[:, cfg_online.train_steps-1:cfg_online.train_steps+cfg_online.M, :, :]
        target = states_dec[:, 1:, :, :]

        output = self.model.module.dec(states_dec, self.model.module.dec.W, self.es, M=cfg_online.M)

        test_mse = mse_loss(output, target).data
        test_acc = edge_accuracy(self.model.module.dec.W, adj)

        return loss, mse, acc, 0, test_mse, test_acc


    def online_train_nri_v3(self, states: Tensor, adj: Tensor, batch_idx: int) -> Tensor:
        """
        NOTE: the model is first optimized in the observed period (i.e., states[:cfg_online.train_steps]) -> optimized loss
              then, predicts the remaining future period (i.e., states[cfg_online.train_steps:]) -> evaluated loss 

        Args:
            states: [batch, step, node, dim], all node states, including historical states and the states to predict
        """
        force_true_edges = False
        if force_true_edges == True:
            self.edges = F.one_hot(adj, num_classes=2).permute(1,0,2) ### force true relations ###
            #self.edges = torch.bernoulli(torch.rand(20,1,2)).cuda()

        # compute the relation distribution (prob) and predict future node states (output)
        self.model.train()
        # skip_first=True: this ensures edges[:,:,1] represents interaction and edges[:,:,0] represents no interaction.
        self.model.module.dec.skip_first = False #True 

        states_enc = states[:, :cfg_online.train_steps, :, :]
        states_dec = states[:, :, :, :]
        #states_dec = states[:, :, :, :]
        target_enc = states_enc[:,1:,:,:]
        target = states_dec[:,1:,:,:]

        """
        NOTE: step1 - predict the relation type using the derivative
        """
        n_atoms = states.shape[2]
        if batch_idx == 0 and force_true_edges == False:
            self.edges = (torch.ones((n_atoms*(n_atoms-1),1,2), requires_grad=True) * 0.5).cuda()
            #self.edges = (torch.rand((n_atoms*(n_atoms-1),1,2), requires_grad=True)).cuda()

        output_enc, new_edges = self.model.module.predict_states(states_enc, self.edges, M=1, target=target_enc, update_z=False)
  
        if force_true_edges == False:
            if new_edges is not None:
                # continuous update
                pred_relations = new_edges.detach()
                pred_relations = torch.mean(pred_relations, dim=1, keepdim=True)
                pred_relations.requires_grad = True
            else:
                relative_target_enc = target_enc.repeat(n_atoms,1,1,1) - target_enc.permute(2,1,0,3).repeat(1,1,n_atoms,1)
                relative_output_enc = output_enc.repeat(n_atoms,1,1,1) - output_enc.permute(2,1,0,3).repeat(1,1,n_atoms,1)

                first_derivative = torch.autograd.grad(F.mse_loss(output_enc, target_enc), self.edges, retain_graph=True)
                #first_derivative = torch.autograd.grad(F.mse_loss(output_enc, target_enc) + F.mse_loss(relative_output_enc, relative_target_enc), self.edges, retain_graph=True)
                #first_derivative = torch.autograd.grad(nll_gaussian(target_enc, output_enc, 5e-5), self.edges, retain_graph=True)

                pred_relations = self.edges - 0.5 * first_derivative[0]
                # binary decision
                # pred_relations = F.one_hot((first_derivative[0][:,:,0] > first_derivative[0][:,:,1]).long(), num_classes=2).float()
                #pred_relations = torch.clamp(pred_relations, min=0, max=1)
                """
                pred_relations_sum = torch.zeros_like(pred_relations).cuda()
                first_derivative_list = torch.zeros_like(first_derivative[0][:,:,1]).repeat(1,1,29)
                for i in range(29):
                    first_derivative = torch.autograd.grad(F.mse_loss(output_enc[:,i,:,:], target_enc[:,i,:,:]), edges, retain_graph=True)
                    temp_pred_relations = F.one_hot((first_derivative[0][:,:,0] > first_derivative[0][:,:,1]).long(), num_classes=2)
                    pred_relations_sum += temp_pred_relations

                    first_derivative_list[0,:,i] = first_derivative[0][:,0,1]
                """

        """
        NOTE: step2 - predict the trajectory using the predicted relation
        """
        if force_true_edges == False:
            self.edges = pred_relations
            #self.edges.requires_grad = True

        output, _ = self.model.module.predict_states(states_dec, self.edges, M=cfg_online.timesteps-cfg_online.train_steps)
        #output, _ = self.model.module.predict_states(states_dec, self.edges, M=10)

        if output.shape[0] == 1:
            # reconstruction loss and the KL-divergence
            loss_nll = nll_gaussian(target, output, 5e-5)
            #relative_target = target.repeat(target.shape[2],1,1,1) - target.permute(2,1,0,3).repeat(1,1,target.shape[2],1)
            #relative_output = output.repeat(output.shape[2],1,1,1) - output.permute(2,1,0,3).repeat(1,1,output.shape[2],1)
            #loss_nll = F.mse_loss(output, target) + F.mse_loss(relative_output, relative_target)
            #loss_nll = F.mse_loss(output, target, reduction='none')
            #loss_nll = loss_nll.mean(dim=(0,1,3)) 
            loss = loss_nll#.mean() + torch.mean(loss_nll)
        else:
            loss_nll = nll_gaussian(target, output[0].unsqueeze(0), 5e-5)
            loss_var = F.mse_loss(output[1:], target.repeat(99,1,1,1), reduction='none').mean(dim=(1,3))
            loss_var = torch.var(loss_var, dim=0)
            loss = loss_nll

        self.optimize(self.opt, loss, retain_graph=False)# * cfg_online.scale)

        if force_true_edges == True:
            pred_relations = self.edges

        mse = mse_loss(output, target).data
        acc = edge_accuracy(pred_relations, adj)
        f1_list = edge_f1(pred_relations, adj)

        plot_trajectories(target[0].permute(1,0,2), output[0].permute(1,0,2), path=self.path+"/plot/"+str(batch_idx)+".png")

        return loss, mse, acc, f1_list


    def online_train_nri_v4(self, states: Tensor, adj: Tensor, batch_idx: int) -> Tensor:
        """
        NOTE: the model is first optimized in the observed period (i.e., states[:cfg_online.train_steps]) -> optimized loss
              then, predicts the remaining future period (i.e., states[cfg_online.train_steps:]) -> evaluated loss 

        Args:
            states: [batch, step, node, dim], all node states, including historical states and the states to predict
        """
        # compute the relation distribution (prob) and predict future node states (output)
        #self.model.module.dec.skip_first = False

        n_atoms = states.shape[2]
        if batch_idx == 0:
            self.edges = nn.Parameter( torch.ones((n_atoms*(n_atoms-1),1,2), requires_grad=True) / 2 )
            self.edges = self.edges.cuda()


        if batch_idx % 10 == 0:
            if batch_idx == 0:
                self.prev_edges = self.edges
                self.delta_edges = 0
            else:
                self.curr_edges = self.edges
                self.delta_edges = torch.abs(self.curr_edges - self.prev_edges).mean()
                self.prev_edges = self.curr_edges

        if self.cmd.adapt_lr == True:
            if self.delta_edges < self.cmd.adapt_threshold:
                self.cmd.edge_lr += self.cmd.adapt_edge_step
            else:
                self.cmd.edge_lr -= self.cmd.adapt_edge_step

            if self.cmd.edge_lr > self.cmd.upper_lr:
                self.cmd.edge_lr = self.cmd.upper_lr
            if self.cmd.edge_lr < self.cmd.lower_lr:
                self.cmd.edge_lr = self.cmd.lower_lr

        target = states[:,1:,:,:]
        output = self.model.module.predict_states(states, self.edges, M=cfg_online.prediction_steps, target=target, update_z=False)

        first_derivative = torch.autograd.grad(F.mse_loss(output, target), self.edges, retain_graph=True)
        if batch_idx >= 0:
            pred_relations = self.edges - self.cmd.edge_lr * first_derivative[0]

        with torch.no_grad():
            pred_relations[pred_relations < 0] = 0
            pred_relations[pred_relations > 1] = 1
            if self.cmd.skip == False:
                pred_relations = pred_relations / (pred_relations.sum(dim=-1, keepdim=True) + 1e-10)
                pred_relations.requires_grad = True

        loss_nll = nll_gaussian(target, output, 5e-5)
        loss = loss_nll
        self.optimize(self.opt, loss * cfg_online.scale)

        self.edges = pred_relations  

        if self.cmd.dyn == "motion":
            mse = mse_loss(output[:,9:], states[:,10:]).data
            mse_t1 = mse_loss(output[:,9], states[:,10]).data
            mse_t5 = mse_loss(output[:,13], states[:,14]).data
            mse_t10 = mse_loss(output[:,18], states[:,19]).data
            mse_list = [mse_t1, mse_t5, mse_t10]
        else:
            mse = mse_loss(output[:,29:], target[:,29:]).data
            mse_t1 = mse_loss(output[:,29], target[:,29]).data
            mse_t2 = mse_loss(output[:,30], target[:,30]).data
            mse_t5 = mse_loss(output[:,33], target[:,33]).data
            mse_t8 = mse_loss(output[:,36], target[:,36]).data
            mse_t10 = mse_loss(output[:,38], target[:,38]).data
            mse_t20 = mse_loss(output[:,48], target[:,48]).data
            mse_t30 = mse_loss(output[:,58], target[:,58]).data
            mse_list = [mse_t1, mse_t10, mse_t20, mse_t30]

        if self.cmd.dyn != "motion" and self.cmd.dyn != "bball":
            acc = edge_accuracy(pred_relations, adj.repeat(pred_relations.size(1),1))
        else:
            acc = 0

        if batch_idx % 1 == 0 and ("motion" not in self.cmd.dyn) and ("bball" not in self.cmd.dyn):
            true_edges = F.one_hot(adj, num_classes=2).permute(1,0,2).repeat(1,self.edges.shape[1],1)
            false_edges = 1 - true_edges
            with torch.no_grad():
                output_true_edges = self.model.module.predict_states(states, true_edges, M=cfg_online.prediction_steps)
                output_false_edges = self.model.module.predict_states(states, false_edges, M=cfg_online.prediction_steps)
            mse_true_edges = mse_loss(output_true_edges[:,29:], states[:,30:]).data
            mse_false_edges = mse_loss(output_false_edges[:,29:], states[:,30:]).data
        else:
            mse_true_edges, mse_false_edges = 0, 0
        mse_list.append(mse_true_edges)
        mse_list.append(mse_false_edges)

        # mse_list.append(mse_t2)
        # mse_list.append(mse_t5)
        # mse_list.append(mse_t8)

        if self.cmd.dyn == "motion":
            plot_motions_edge_static(pred_relations.permute(1,0,2), self.rel_rec, self.rel_send, path=self.path + "/plot/edge"+str(batch_idx)+".png")
            plot_motions(output[:,9:].squeeze().permute(1,0,2), states[:,10:].squeeze().permute(1,0,2), path=self.path + "/plot/"+str(batch_idx)+".png")
            np.savetxt(self.path + "/edge/edge_pred" + str(batch_idx) + ".csv", np.array(vec2mat(pred_relations[:,:,1].permute(1,0), self.cmd.size).detach().cpu()))

        return loss, mse, mse_list, acc


    def train_nri(self, states: Tensor) -> Tensor:
        """
        Args:
            states: [batch, step, node, dim], all node states, including historical states and the states to predict
        """
        # compute the relation distribution (prob) and predict future node states (output)
        output, prob = self.model(states, states, p=True, M=cfg.M, tosym=cfg.sym)
        prob = prob.transpose(0, 1).contiguous()
        # reconstruction loss and the KL-divergence
        loss_nll = nll_gaussian(output, states[:, 1:], 5e-5)
        loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
        loss = loss_nll + loss_kl
        # impose the soft symmetric contraint by adding a regularization term
        if self.cmd.reg > 0:
            # transpose the relation distribution
            prob_hat = transpose(prob, self.size)
            loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
            loss = loss + loss_sym * self.cmd.reg
        self.optimize(self.opt, loss * cfg.scale)

        # choice for the evaluation metric, adding the regularization term or not
        # when the penalty factor is large, it may be misleading to add the regularization term
        if self.cmd.no_reg:
            loss = loss_nll + loss_kl
        return loss

    def test(self, name: str, M: int):
        """
        Evaluate related metrics to measure the model performance.
        The biggest difference between this function and evalute() is that, the mses are evaluated at each step.

        Args:
            name: 'train' / 'val' / 'test'
            M: number of steps to predict

        Return:
            mse_multi: mse at each step
        """
        """
        acc: accuracy of relation reconstruction
        mses: mean square error over all steps
        rate: rate of assymmetry
        ratio: relative root mean squared error
        sparse: rate of sparsity in terms of the first type of edge
        losses: loss_nll + loss_kl (+ loss_reg) 
        mse_multi: mse at each step
        """
        acc, mses, rate, ratio, sparse, losses, mse_multi = [], [], [], [], [], [], []
        data = self.load_data(self.data[name], self.batch_size)
        N = 0.
        with torch.no_grad():
            for adj, states in data:
                if cfg.gpu:
                    adj = adj.cuda()
                    states = states.cuda()
                states_enc = states[:, :cfg.train_steps, :, :]
                states_dec = states[:, -cfg.train_steps:, :, :]
                target = states_dec[:, 1:]
                
                #print("[test1] states, states_enc, states_dec, target: ", states.shape, states_enc.shape, states_dec.shape, target.shape)

                output, prob = self.model(states_enc, states_dec, hard=True, p=True, M=M, tosym=cfg.sym)
                prob = prob.transpose(0, 1).contiguous()

                scale = len(states) / self.batch_size
                N += scale

                # use loss as the validation metric
                loss_nll = nll_gaussian(target, output, 5e-5)
                loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
                loss = loss_nll + loss_kl
                if self.cmd.reg > 0 and not self.cmd.no_reg:
                    prob_hat = transpose(prob, self.size)
                    loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
                    loss = loss + loss_sym * self.cmd.reg

                # scale all metrics to match the batch size
                loss = loss * scale
                losses.append(loss)

                mses.append(scale * mse_loss(output, target).data)
                ratio.append(scale * (((output - target) ** 2).sum(-1).sqrt() / (target ** 2).sum(-1).sqrt()).mean())
                acc.append(scale * edge_accuracy(prob, adj))
                _, p = prob.max(-1)
                rate.append(scale * asym_rate(p.t(), self.size))
                sparse.append(prob.max(-1)[1].float().mean() * scale)

                states_dec = states[:, cfg.train_steps:cfg.train_steps+M+1, :, :]
                target = states_dec[:, 1:]
                
                print("[test2] states, states_enc, states_dec, target: ", states.shape, states_enc.shape, states_dec.shape, target.shape)
                input("here")

                output, prob = self.model(states_enc, states_dec, hard=True, p=True, M=M, tosym=cfg.sym)
                prob = prob.transpose(0, 1).contiguous()
                mse = ((output - target) ** 2).mean(dim=(0, 2, -1))
                mse *= scale
                mse_multi.append(mse)
        loss = sum(losses) / N
        mses = sum(mses) / N
        mse_multi = sum(mse_multi) / N
        acc = sum(acc) / N
        rate = sum(rate) / N
        ratio = sum(ratio) / N
        sparse = sum(sparse) / N
        self.log.info('{} M {:02d} mse {:.3e} acc {:.4f} _acc {:.4f} rate {:.4f} ratio {:.4f} sparse {:.4f}'.format(
                name, M, mses, acc, 1 - acc, rate, ratio, sparse))
        msteps = ','.join(['{:.3e}'.format(i) for i in mse_multi])
        self.log.info(msteps)
        return mse_multi

    def evalate(self, test, M: int):
        """
        Evaluate related metrics to monitor the training process.

        Args:
            test: data set to be evaluted
            M: number of steps to predict

        Return:
            loss: loss_nll + loss_kl (+ loss_reg) 
            mse: mean square error over all steps
            acc: accuracy of relation reconstruction
            rate: rate of assymmetry
            ratio: relative root mean squared error
            sparse: rate of sparsity in terms of the first type of edge
        """
        acc, mse, rate, ratio, sparse, losses = [], [], [], [], [], []
        data = self.load_data(test, self.batch_size)
        N = 0.
        with torch.no_grad():
            for adj, states in data:
                if cfg.gpu:
                    adj = adj.cuda()
                    states = states.cuda()
                states_enc = states[:, :cfg.train_steps, :, :]
                states_dec = states[:, -cfg.train_steps:, :, :]
                target = states_dec[:, 1:]
                
                #print("[elevate] states, states_enc, states_dec, target: ", states.shape, states_enc.shape, states_dec.shape, target.shape)
                #input("here")

                output, prob = self.model(states_enc, states_dec, hard=True, p=True, M=M, tosym=cfg.sym)
                prob = prob.transpose(0, 1).contiguous()

                scale = len(states) / self.batch_size
                N += scale

                # use loss as the validation metric
                loss_nll = nll_gaussian(target, output, 5e-5)
                loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
                loss = loss_nll + loss_kl
                if self.cmd.reg > 0 and not self.cmd.no_reg:
                    prob_hat = transpose(prob, self.size)
                    loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
                    loss = loss + loss_sym * self.cmd.reg
                # scale all metrics to match the batch size
                loss = loss * scale
                losses.append(loss)

                mse.append(scale * mse_loss(output, target).data)
                ratio.append(scale * (((output - target) ** 2).sum(-1).sqrt() / (target ** 2).sum(-1).sqrt()).mean())
                acc.append(scale * edge_accuracy(prob, adj))
                _, p = prob.max(-1)
                rate.append(scale * asym_rate(p.t(), self.size))
                sparse.append(prob.max(-1)[1].float().mean() * scale)
        loss = sum(losses) / N
        mse = sum(mse) / N
        acc = sum(acc) / N
        rate = sum(rate) / N
        ratio = sum(ratio) / N
        sparse = sum(sparse) / N
        return loss, mse, acc, rate, ratio, sparse
