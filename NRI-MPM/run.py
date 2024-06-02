import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import config as cfg
import config_online as cfg_online
from instructors.XNRI import XNRIIns
from instructors.XNRI_enc import XNRIENCIns
from instructors.XNRI_dec import XNRIDECIns
from argparse import ArgumentParser
from utils.general import read_pickle
from models.encoder import AttENC, RNNENC, GNNENC
from models.decoder import GNNDEC, RNNDEC, AttDEC
from models.nri import NRIModel
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from itertools import permutations
from generate.load import load_kuramoto, load_nri, online_load_nri, CmuMotionData


def init_args():
    parser = ArgumentParser()
    parser.add_argument('--dyn', type=str, default='springs', 
    help='Type of dynamics: springs, charged, springs_var, charged_var, mixed.')
    parser.add_argument('--size', type=int, default=10, help='Number of particles.')
    parser.add_argument('--dim', type=int, default=4, help='Dimension of the input states.')
    parser.add_argument('--epochs', type=int, default=1,help='Number of training epochs. 0 for testing.')
    parser.add_argument('--reg', type=float, default=0, help='Penalty factor for the symmetric prior.')
    parser.add_argument('--batch', type=int, default=1, help='Batch size.')
    parser.add_argument('--skip', action='store_true', default=False, help='Skip the last type of edge.')
    parser.add_argument('--no_reg', action='store_true', default=False, help='Omit the regularization term when using the loss as an validation metric.')
    parser.add_argument('--sym', action='store_true', default=False, help='Hard symmetric constraint.')
    parser.add_argument('--reduce', type=str, default='cnn', help='Method for relation embedding, mlp or cnn.')
    parser.add_argument('--enc', type=str, default='RNNENC', help='Encoder.')
    parser.add_argument('--dec', type=str, default='RNNDEC', help='Decoder.')
    parser.add_argument('--scheme', type=str, default='both', help='Training schemes: both, enc or dec.')
    parser.add_argument('--load_path', type=str, default='', help='Where to load a pre-trained model.')
    parser.add_argument('--seed', type=int, default=43, help='Random seed.')
    parser.add_argument('--online', type=bool, default=True)
    parser.add_argument('--trajectory-flip', action='store_true', default=True, help="enable Trajectory Mirror")
    parser.add_argument('--edge_lr', type=float, default=20,
                        help='initial learning rate for interaction graph')
    parser.add_argument('--adapt_lr', action='store_true', default=True, help='enable AdaRelation')
    parser.add_argument('--upper_lr', type=float, default=50,
                        help='upper bound of learning rate for interaction graph')
    parser.add_argument('--lower_lr', type=float, default=20,
                        help='lower bound of learning rate for interaction graph')
    parser.add_argument('--adapt_edge_step', type=float, default=1)
    parser.add_argument('--adapt_threshold', type=float, default=0.05)
    return parser.parse_args()


def load_data(args):
    if args.online == True:
        if args.dyn == "motion":
            # CMU MoCap data (walking motion)
            args.data_path = '../data/motion/processed/35/'
            params = {}
            train_data = CmuMotionData('cmu', args.data_path, 'train', params)
            train_loader = DataLoader(train_data, batch_size=1, shuffle=False, drop_last=True)
            args.size = 31
            args.dim = 6
            args.trajectory_flip = False
            args.skip = True
            cfg_online.timesteps = 20
            cfg_online.prediction_steps = 10
            cfg_online.M = 10

            return train_loader

        else:
            if args.dyn == "springs":
                # evolving interaction + fixed parameter (k)
                args.data_path = '../data/evolving_interaction/springs10_interaction0.1_90k/'
                edges = np.load(args.data_path + 'edges_train_springs' + str(args.size) + '.npy')
                loc = np.load(args.data_path + 'loc_train_springs' + str(args.size) + '.npy').transpose(0,1,3,2)
                vel = np.load(args.data_path + 'vel_train_springs' + str(args.size) + '.npy').transpose(0,1,3,2)
                features = np.concatenate([loc, vel], axis=-1)
                args.skip = True
                return {'train': (edges, features)}

            elif args.dyn == "springs_var":
                # evolving interaction + evolving parameter (k)
                args.data_path = '../data/evolving_parameter/springs10_interaction_var'
                edges = np.load(args.data_path + 'edges_train_springs' + str(args.size) + '.npy')
                loc = np.load(args.data_path + 'loc_train_springs' + str(args.size) + '.npy').transpose(0,1,3,2)
                vel = np.load(args.data_path + 'vel_train_springs' + str(args.size) + '.npy').transpose(0,1,3,2)
                features = np.concatenate([loc, vel], axis=-1)
                args.skip = True
                return {'train': (edges, features)}
            
            elif args.dyn == "charged":
                # evolving interaction + fixed parameter (k_e)
                args.data_path = '../data/evolving_interaction/charged10_interaction1.0_90k/'
                edges = np.load(args.data_path + 'edges_train_charged' + str(args.size) + '.npy')
                loc = np.load(args.data_path + 'loc_train_charged' + str(args.size) + '.npy').transpose(0,1,3,2)
                vel = np.load(args.data_path + 'vel_train_charged' + str(args.size) + '.npy').transpose(0,1,3,2)
                features = np.concatenate([loc, vel], axis=-1)
                return {'train': (edges, features)}

            elif args.dyn == "charged_var":
                # evolving interaction + evolving parameter (k_e)
                args.data_path = '../data/evolving_parameter/charged10_interaction_var/'
                edges = np.load(args.data_path + 'edges_train_charged' + str(args.size) + '.npy')
                loc = np.load(args.data_path + 'loc_train_charged' + str(args.size) + '.npy').transpose(0,1,3,2)
                vel = np.load(args.data_path + 'vel_train_charged' + str(args.size) + '.npy').transpose(0,1,3,2)
                features = np.concatenate([loc, vel], axis=-1)
                return {'train': (edges, features)}

            elif args.dyn == "mixed":
                # evolving interaction + switching dynamics (fixed paremeter)
                args.data_path = '../data/evolving_dynamics/mixed/'
                edges = np.load(args.data_path + 'edges_train_mix' + str(args.size) + '.npy')
                loc = np.load(args.data_path + 'loc_train_mix' + str(args.size) + '.npy').transpose(0,1,3,2)
                vel = np.load(args.data_path + 'vel_train_mix' + str(args.size) + '.npy').transpose(0,1,3,2)
                features = np.concatenate([loc, vel], axis=-1)
                return {'train': (edges, features)}

    else:
        path = 'data/{}/{}.pkl'.format(args.dyn, args.size)
        train, val, test = read_pickle(path)
        data = {'train': train, 'val': val, 'test': test}
        # each data (train, val, test) has key (interaction graph; (5,5)) and values (features; (99,5,4))
        return data


def run():
    args = init_args()
    if args.online == True:
        args.batch = 1
        args.epochs = 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    cfg.init_args(args)

    # load data
    # NOTE: in Online setting, training batch should not be shuffled (see train() in XNRI.py) !
    data = load_data(args)
    if args.online == True:
        if args.dyn == "motion":
            es = np.array(list(permutations(range(args.size), 2))).T
            data = data
        else:
            data, es = online_load_nri(data, args.size)
    else:
        if args.dyn == 'kuramoto':
            data, es, _ = load_kuramoto(data, args.size)
        else:
            data, es, _ = load_nri(data, args.size)

    if args.online == True:
        dim = args.dim if args.reduce == 'cnn' else args.dim * cfg_online.train_steps
    else:
        dim = args.dim if args.reduce == 'cnn' else args.dim * cfg.train_steps
    encs = {
        'GNNENC': GNNENC,
        'RNNENC': RNNENC,
        'AttENC': AttENC,
    }
    decs = {
        'GNNDEC': GNNDEC,
        'RNNDEC': RNNDEC,
        'AttDEC': AttDEC,
    }
    if args.online == True:
        encoder = encs[args.enc](dim, cfg_online.n_hid, cfg_online.edge_type, reducer=args.reduce)
        decoder = decs[args.dec](args.dim, cfg_online.edge_type, cfg_online.n_hid, cfg_online.n_hid, cfg_online.n_hid, skip_first=args.skip)
    else:
        encoder = encs[args.enc](dim, cfg.n_hid, cfg.edge_type, reducer=args.reduce)
        decoder = decs[args.dec](args.dim, cfg.edge_type, cfg.n_hid, cfg.n_hid, cfg.n_hid, skip_first=args.skip)
    model = NRIModel(encoder, decoder, es, args.size)
    if args.load_path:
        name = 'logs/{}/best.pth'.format(args.load_path)
        model.load_state_dict(torch.load(name)) 
    model = DataParallel(model)

    if cfg.gpu:
        model = model.cuda()
    if args.scheme == 'both':
        # Normal training.
        ins = XNRIIns(model, data, es, args)
    elif args.scheme == 'enc':
        # Only train the encoder.
        ins = XNRIENCIns(model, data, es, args)
    elif args.scheme == 'dec':
        # Only train the decoder.
        ins = XNRIDECIns(model, data, es, args)
    else:
        raise NotImplementedError('training scheme: both, enc or dec')

    #ins.train()
    ins.single_batch_train()


if __name__ == "__main__":
    for _ in range(cfg.rounds):
        run()
