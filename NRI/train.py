from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import time
import argparse
import pickle
import datetime

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=43, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=10,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='onlinernn',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_springs',
                    help='Suffix for training data (e.g. "_springs", "_springs_var", "_charged", "charged_var", "_mixed".')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=60,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=30, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument('--online', action='store_true', default=True,
                    help='NRI-based method -> False')
parser.add_argument('--trajectory-flip', action='store_true', default=True)
parser.add_argument('--edge_lr', type=float, default=100,
                    help='initial learning rate for interaction graph')
parser.add_argument('--adapt_lr', action='store_true', default=True)
parser.add_argument('--upper_lr', type=float, default=200,
                    help='upper bound of learning rate for interaction graph')
parser.add_argument('--lower_lr', type=float, default=100,
                    help='lower bound of learning rate for interaction graph')
parser.add_argument('--edge_lr_step', type=float, default=1)
parser.add_argument('--adapt_threshold', type=float, default=0.05)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/{}_exp{}/'.format(args.save_folder, args.suffix.replace("_", "") + str(args.num_atoms), timestamp)
    os.mkdir(save_folder)
    os.mkdir(save_folder+"plot/")
    os.mkdir(save_folder+"edge/")
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

if "motion" in args.suffix:
    data_path = '../data/motion/processed/35/'
    params = {}
    train_data = CmuMotionData('cmu', data_path, 'train', params)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, drop_last=True)
    args.num_atoms = 31
    args.timesteps = 20
    args.prediction_steps = 10
    args.dims = 6
    args.trajectory_flip = False
    args.skip_first = True

else:
    if "_springs" == args.suffix:
        # evolving interaction + fixed parameter (k)
        data_path = "../data/evolving_interaction/springs10_interaction0.1_90k"
        train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
            batch_size=args.batch_size, path=data_path, suffix=args.suffix + str(args.num_atoms))
        args.skip_first = True

    elif "_springs_var" == args.suffix:
        # evolving interaction + evolving parameter (k)
        data_path = '../data/evolving_parameter/springs10_interaction_var'
        train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
            batch_size=args.batch_size, path=data_path, suffix=args.suffix.replace("_var", "") + str(args.num_atoms))
        args.skip_first = True

    elif "_charged" == args.suffix:
        # evolving interaction + fixed parameter (k_e)
        data_path = "../data/evolving_interaction/charged10_interaction1.0_90k"
        train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
            batch_size=args.batch_size, path=data_path, suffix=args.suffix + str(args.num_atoms))

    elif "_charged_var" == args.suffix:
        # evolving interaction + evolving parameter (k_e)
        data_path = '../data/evolving_parameter/charged10_interaction_var'
        train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
            batch_size=args.batch_size, path=data_path, suffix=args.suffix.replace("_var", "") + str(args.num_atoms))

    elif "_mixed" == args.suffix:
        # evolving interaction + switching dynamics (fixed paremeter)
        data_path = '../data/evolving_dynamics/mixed'
        train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
            batch_size=args.batch_size, path=data_path, suffix=args.suffix.replace("ed", "") + str(args.num_atoms))

# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.encoder == 'mlp':
    encoder = MLPEncoder(int(args.timesteps * args.dims / 2), args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)

elif args.encoder == 'cnn':
    encoder = CNNEncoder(args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)

if args.decoder == 'mlp':
    decoder = MLPDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'rnn':
    decoder = RNNDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'onlinernn':
    decoder = OnlineRNNDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first,
                         n_atoms=args.num_atoms)
elif args.decoder == 'sim':
    decoder = SimulationDecoder(loc_max, loc_min, vel_max, vel_min, args.suffix)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)

if args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    if args.online == False:
        encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)

def vec2mat(vector, size):
    # vector size should be (20,1)
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

def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []
    mse_test = []
    var_traj = []
    
    acc_list = []

    init_edges = True

    encoder.train()
    decoder.train()
    scheduler.step()

    for batch_idx, (data, relations) in enumerate(train_loader):
        if "motion" in args.suffix:
            data = data.permute(0,2,1,3)
            relations = torch.zeros((1, 930, 2))

        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data), Variable(relations)

        optimizer.zero_grad()

        if args.online == False:
            logits = encoder(data, rel_rec, rel_send) # (batch=1, num edges=20, num edge types=2)
            edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard) # (batch=1, num edges=20, num edge types=2)
            prob = my_softmax(logits, -1) # (batch=1, num edges=20, num edge types=2)
            #print("[bef] logits, edges, prob: ", logits.shape, edges.shape, prob.shape)

            target = data[:, :, 1:, :]

            if args.decoder == 'rnn':
                output = decoder(data, edges, rel_rec, rel_send, 100,
                                burn_in=True,
                                burn_in_steps=args.timesteps - args.prediction_steps, test=True)
            else:
                output, new_edges = decoder(data, edges, rel_rec, rel_send,
                                args.prediction_steps, target=target, test=True)

            loss_nll = nll_gaussian(output, target, args.var)
            if args.prior:
                loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
            else:
                loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                                  args.edge_types)
            loss = loss_nll + loss_kl
            loss.backward()
            optimizer.step()

            acc = edge_accuracy(logits, relations)
            acc_train.append(acc)

            mse = F.mse_loss(output, target)
            mse_train.append(mse.item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())

        else:
            if "motion" in args.suffix:
                num_iters = 4
            else:
                num_iters = 2999

            if init_edges == True:
                acc = 0
                edges = nn.Parameter( torch.ones((1, args.num_atoms*(args.num_atoms-1), 2), requires_grad=True) / 2 )
                edges = edges.cuda()

                edges_low_lr = nn.Parameter( torch.ones((1, args.num_atoms*(args.num_atoms-1), 2), requires_grad=True) / 2 )
                edges_low_lr = edges.cuda()
                init_edges = False


            for i in range(num_iters):
                force_true_edge = False

                start_idx = i * args.prediction_steps
                end_idx = start_idx + args.timesteps

                temp_relations = relations#[:,start_idx,:]
                temp_data = data[:,:,start_idx:end_idx,:]
                temp_data_aug = temp_data.clone()

                if args.trajectory_flip == True:
                    temp_data_aug = temp_data_aug.repeat(4,1,1,1)
                    for flip in range(4):
                        if flip == 0: # same
                            temp_data_aug[flip,:,:,(0,2)] = temp_data_aug[flip,:,:,(0,2)]
                            temp_data_aug[flip,:,:,(1,3)] = temp_data_aug[flip,:,:,(1,3)]
                        elif flip == 1: # flip x-axis
                            temp_data_aug[flip,:,:,(0,2)] = temp_data_aug[flip,:,:,(0,2)] * -1
                            temp_data_aug[flip,:,:,(1,3)] = temp_data_aug[flip,:,:,(1,3)]
                        elif flip == 2: # flip y-axis
                            temp_data_aug[flip,:,:,(0,2)] = temp_data_aug[flip,:,:,(0,2)]
                            temp_data_aug[flip,:,:,(1,3)] = temp_data_aug[flip,:,:,(1,3)] * -1
                        elif flip == 3: # flip x and y-axis
                            temp_data_aug[flip,:,:,(0,2)] = temp_data_aug[flip,:,:,(0,2)] * -1
                            temp_data_aug[flip,:,:,(1,3)] = temp_data_aug[flip,:,:,(1,3)] * -1


                if "motion" not in args.suffix:
                    true_edges = F.one_hot(temp_relations, num_classes=2).float() ### force true relations ###
                    #true_edges = torch.bernoulli(torch.rand(temp_relations.shape[0],temp_relations.shape[1],2)).cuda().float()
                    true_edges.requires_grad = True
                else:
                    true_edges = torch.bernoulli(torch.rand(temp_relations.shape[0],temp_relations.shape[1],2)).cuda().float()
                    #true_edges = torch.ones_like(true_edges) #/ 2
                    true_edges.requires_grad = True

                if (i + num_iters * batch_idx) % 10 == 0:
                    if (i + num_iters * batch_idx) == 0:
                        prev_edges = edges
                        delta_edges = 0
                    else:
                        curr_edges = edges
                        delta_edges = torch.abs(curr_edges - prev_edges).mean()
                        prev_edges = curr_edges


                if args.adapt_lr == True:
                    if delta_edges < args.adapt_threshold:
                        args.edge_lr += args.edge_lr_step
                    else:
                        args.edge_lr -= args.edge_lr_step

                    if args.edge_lr > args.upper_lr:
                        args.edge_lr = args.upper_lr
                    if args.edge_lr < args.lower_lr:
                        args.edge_lr = args.lower_lr

                target = temp_data_aug[:,:,1:,:].squeeze()

                if "rnn" in args.decoder:
                    if force_true_edge == True:
                        output, new_edges = decoder(temp_data_aug, true_edges, rel_rec, rel_send,
                                        args.prediction_steps, test=True, target=target, burn_in_steps=args.timesteps-args.prediction_steps)
                    else:
                        output, new_edges = decoder(temp_data_aug, edges, rel_rec, rel_send,
                                        args.prediction_steps, test=True, target=target, burn_in_steps=args.timesteps-args.prediction_steps)
                elif "mlp" in args.decoder:
                    if force_true_edge == True:
                        output = decoder(temp_data_aug, true_edges, rel_rec, rel_send,
                                        args.prediction_steps, test=True)
                    else:
                        output = decoder(temp_data_aug, edges, rel_rec, rel_send,
                                        args.prediction_steps, test=True)

                if force_true_edge == False:
                    edges_loss = F.mse_loss(output.squeeze(), target)
                    first_derivative = torch.autograd.grad(edges_loss, edges, retain_graph=True)
                    pred_relations = edges - args.edge_lr * first_derivative[0]

                    with torch.no_grad():
                        pred_relations[pred_relations < 0] = 0
                        pred_relations[pred_relations > 1] = 1
                        if args.skip_first == False:
                            pred_relations = pred_relations / (pred_relations.sum(dim=-1, keepdim=True) + 1e-10)
                            #pred_relations = pred_relations / (pred_relations.max() + 1e-10)
                            pred_relations.requires_grad = True

                    edges = pred_relations # new edges
                    if "motion" not in args.suffix:
                        acc = edge_accuracy(edges, temp_relations)

                loss_nll = nll_gaussian(output, target, args.var)
                loss = loss_nll

                # print("output, target: ", output.shape, target.shape)
                # input("here")
                if "motion" in args.suffix:
                    mse_t = F.mse_loss(output[:,:,9:,:], target[:,9:,:])
                    mse_t1 = F.mse_loss(output[:,:,9,:], target[:,9,:])
                    mse_t5 = F.mse_loss(output[:,:,13,:], target[:,13,:])
                    mse_t10 = F.mse_loss(output[:,:,18,:], target[:,18,:])
                else:
                    mse_t = F.mse_loss(output[:,:,29:,:], target[:,:,29:,:])
                    mse_t1 = F.mse_loss(output[:,:,29,:], target[:,:,29,:])
                    mse_t2 = F.mse_loss(output[:,:,30,:], target[:,:,30,:])
                    mse_t5 = F.mse_loss(output[:,:,33,:], target[:,:,33,:])
                    mse_t8 = F.mse_loss(output[:,:,36,:], target[:,:,36,:])
                    mse_t10 = F.mse_loss(output[:,:,38,:], target[:,:,38,:])
                    mse_t20 = F.mse_loss(output[:,:,48,:], target[:,:,48,:])
                    mse_t30 = F.mse_loss(output[:,:,58,:], target[:,:,58,:])

                if batch_idx % 1 == 0 and ("motion" not in args.suffix):
                    true_edges = F.one_hot(temp_relations, num_classes=2).repeat(edges.shape[0],1,1)
                    false_edges = 1 - true_edges
                    with torch.no_grad():
                        output_true_edges, new_edges = decoder(temp_data_aug, true_edges, rel_rec, rel_send,
                                        args.prediction_steps, test=True, target=target, burn_in_steps=args.timesteps-args.prediction_steps)
                        output_false_edges, new_edges = decoder(temp_data_aug, false_edges, rel_rec, rel_send,
                                        args.prediction_steps, test=True, target=target, burn_in_steps=args.timesteps-args.prediction_steps)
                    mse_true_edges = F.mse_loss(output_true_edges[:,:,29:,:], target[:,:,29:,:]).data
                    mse_false_edges = F.mse_loss(output_false_edges[:,:,29:,:], target[:,:,29:,:]).data
                else:
                    mse_true_edges, mse_false_edges = 0, 0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if "motion" in args.suffix:
                    plot_motions_edge_static(edges, rel_rec, rel_send, path=save_folder+"plot/edge"+str(i + num_iters * batch_idx)+".png")
                    plot_motions(output.squeeze(), target, path=save_folder+"plot/"+str(i + num_iters * batch_idx)+".png")
                    np.savetxt(save_folder + "edge/edge_pred" + str(i+num_iters*batch_idx) + ".csv", np.array(vec2mat(edges[:,:,1], args.num_atoms).detach().cpu()))
                else:
                    pass
                    #plot_trajectories(output.squeeze(), target, path=save_folder+"plot/"+str(i + num_iters * batch_idx)+".png")
   
                if "motion" in args.suffix:
                    print('Batch: {:04d}'.format(i + num_iters * batch_idx),
                        'nll_loss: {:.10f}'.format(float(loss.item())),
                        'mse_pred: {:.12f}'.format(float(mse_t.item())),
                        'mse_t1: {:.12f}'.format(float(mse_t1.item())),
                        'mse_t5: {:.12f}'.format(float(mse_t5.item())),
                        'mse_t10: {:.12f}'.format(float(mse_t10.item())),
                        'edge_lr: {:.10f}'.format(float(args.edge_lr)),
                        'd_edges: {:.5e}'.format(float(delta_edges)),
                        'time: {:.4f}s'.format(time.time() - t), file=log)
                else:
                    print('Batch: {:04d}'.format(i + num_iters * batch_idx),
                        'nll_loss: {:.10f}'.format(float(loss.item())),
                        'mse_pred: {:.12f}'.format(float(mse_t.item())),
                        'mse_t1: {:.12f}'.format(float(mse_t1.item())),
                        'mse_t2: {:.12f}'.format(float(mse_t2.item())),
                        'mse_t5: {:.12f}'.format(float(mse_t5.item())),
                        'mse_t8: {:.12f}'.format(float(mse_t8.item())),
                        'mse_t10: {:.12f}'.format(float(mse_t10.item())),
                        'mse_t20: {:.12f}'.format(float(mse_t20.item())),
                        'mse_t30: {:.12f}'.format(float(mse_t30.item())),
                        'acc: {:.10f}'.format(float(acc)),
                        'edge_lr: {:.10f}'.format(float(args.edge_lr)),
                        'd_edges: {:.5e}'.format(float(delta_edges)),
                        'mse_true {:.12f}'.format(float(mse_true_edges.item())),
                        'mse_false {:.12f}'.format(float(mse_false_edges.item())),
                        'time: {:.4f}s'.format(time.time() - t), file=log)


        if args.online == False:
            print('Batch: {:04d}'.format(batch_idx),
                'nll_train: {:.10f}'.format(float(loss.item())),
                'kl_train: {:.10f}'.format(float(loss_kl.item())),
                'mse_train: {:.12f}'.format(float(mse.item())),
                'acc_train: {:.10f}'.format(float(acc)),
                'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()


    if True: #args.batch_size == 1: # online learning
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Last model, saving...')
        log.flush()

        return np.mean(nll_train)
    else:
        nll_val = []
        acc_val = []
        kl_val = []
        mse_val = []

        encoder.eval()
        decoder.eval()
        for batch_idx, (data, relations) in enumerate(valid_loader):
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()
            data, relations = Variable(data, volatile=True), Variable(
                relations, volatile=True)

            logits = encoder(data, rel_rec, rel_send)
            edges = gumbel_softmax(logits, tau=args.temp, hard=True)
            prob = my_softmax(logits, -1)

            # validation output uses teacher forcing
            output = decoder(data, edges, rel_rec, rel_send, 1)

            target = data[:, :, 1:, :]
            loss_nll = nll_gaussian(output, target, args.var)
            loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

            acc = edge_accuracy(logits, relations)
            acc_val.append(acc)

            mse_val.append(F.mse_loss(output, target).data[0])
            nll_val.append(loss_nll.data[0])
            kl_val.append(loss_kl.data[0])

        print('Epoch: {:04d}'.format(epoch),
            'nll_train: {:.10f}'.format(np.mean(nll_train)),
            'kl_train: {:.10f}'.format(np.mean(kl_train)),
            'mse_train: {:.10f}'.format(np.mean(mse_train)),
            'acc_train: {:.10f}'.format(np.mean(acc_train)),
            'nll_val: {:.10f}'.format(np.mean(nll_val)),
            'kl_val: {:.10f}'.format(np.mean(kl_val)),
            'mse_val: {:.10f}'.format(np.mean(mse_val)),
            'acc_val: {:.10f}'.format(np.mean(acc_val)),
            'time: {:.4f}s'.format(time.time() - t))
        if args.save_folder and np.mean(nll_val) < best_val_loss:
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print('Best model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                'nll_train: {:.10f}'.format(np.mean(nll_train)),
                'kl_train: {:.10f}'.format(np.mean(kl_train)),
                'mse_train: {:.10f}'.format(np.mean(mse_train)),
                'acc_train: {:.10f}'.format(np.mean(acc_train)),
                'nll_val: {:.10f}'.format(np.mean(nll_val)),
                'kl_val: {:.10f}'.format(np.mean(kl_val)),
                'mse_val: {:.10f}'.format(np.mean(mse_val)),
                'acc_val: {:.10f}'.format(np.mean(acc_val)),
                'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()
        return np.mean(nll_val)


def test():
    acc_test = []
    nll_test = []
    kl_test = []
    mse_test = []
    tot_mse = 0
    counter = 0

    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))
    for batch_idx, (data, relations) in enumerate(test_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, volatile=True), Variable(
            relations, volatile=True)

        assert (data.size(2) - args.timesteps) >= args.timesteps

        data_encoder = data[:, :, :args.timesteps, :].contiguous()
        data_decoder = data[:, :, -args.timesteps:, :].contiguous()

        logits = encoder(data_encoder, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=args.temp, hard=True)

        prob = my_softmax(logits, -1)

        output = decoder(data_decoder, edges, rel_rec, rel_send, 1)

        target = data_decoder[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, args.var)
        loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

        acc = edge_accuracy(logits, relations)
        acc_test.append(acc)

        mse_test.append(F.mse_loss(output, target).data[0])
        nll_test.append(loss_nll.data[0])
        kl_test.append(loss_kl.data[0])

        # For plotting purposes
        if args.decoder == 'rnn':
            if args.dynamic_graph:
                output = decoder(data, edges, rel_rec, rel_send, 100,
                                 burn_in=True, burn_in_steps=args.timesteps,
                                 dynamic_graph=True, encoder=encoder,
                                 temp=args.temp)
            else:
                output = decoder(data, edges, rel_rec, rel_send, 100,
                                 burn_in=True, burn_in_steps=args.timesteps)
            output = output[:, :, args.timesteps:, :]
            target = data[:, :, -args.timesteps:, :]
        else:
            data_plot = data[:, :, args.timesteps:args.timesteps + 21,
                        :].contiguous()
            output = decoder(data_plot, edges, rel_rec, rel_send, 20)
            target = data_plot[:, :, 1:, :]

        mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
        tot_mse += mse.data.cpu().numpy()
        counter += 1

    mean_mse = tot_mse / counter
    mse_str = '['
    for mse_step in mean_mse[:-1]:
        mse_str += " {:.12f} ,".format(mse_step)
    mse_str += " {:.12f} ".format(mean_mse[-1])
    mse_str += ']'

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(nll_test)),
          'kl_test: {:.10f}'.format(np.mean(kl_test)),
          'mse_test: {:.10f}'.format(np.mean(mse_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)))
    print('MSE: {}'.format(mse_str))
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.10f}'.format(np.mean(nll_test)),
              'kl_test: {:.10f}'.format(np.mean(kl_test)),
              'mse_test: {:.10f}'.format(np.mean(mse_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)),
              file=log)
        print('MSE: {}'.format(mse_str), file=log)
        log.flush()


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


print("args: ", args, file=log)
print("data: ", data_path, file=log)
print("\n[info] Model named parameters", file=log)
for name, params in encoder.named_parameters():
    print("Encoder name: ", name, params.shape, file=log)
enc_num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
for name, params in decoder.named_parameters():
    print("Decoder name: ", name, params.shape, file=log)
dec_num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print('[Info] Number of Trainable Parameters in Encoder: {}'.format(enc_num_params), file=log)
print('[Info] Number of Trainable Parameters in Decoder: {}\n'.format(dec_num_params), file=log)
log.flush()

# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()

# test()
# if log is not None:
#     print(save_folder)
#     log.close()
