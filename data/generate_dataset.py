# This code is largely based on NRI paper:
# "Neural relational inference for interacting systems." by Thomas Kipf*, Ethan Fetaya*, Kuan-Chieh Wang, Max Welling, Richard Zemel.
# https://github.com/ethanfetaya/NRI/tree/master/data

from synthetic_sim import ChargedParticlesSim, SpringSim
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='springs',
                    help='What simulation to generate.')
parser.add_argument('--mode', type=str, default='interaction',
                    help='What to evolve in simulation. either interaction or parameter')
parser.add_argument('--num-train', type=int, default=10,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=100,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=100,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=9100,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=10000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=10,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--edge_sample-freq', type=int, default=1,
                    help='How often to create new interaction.')

args = parser.parse_args()

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_springs'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_charged'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls)
np.random.seed(args.seed)

print(suffix)

def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges_all = list()


    interaction_list = [0.137, 0.195, 0.173, 0.160, 0.116, 0.116, 0.106, 0.187, 0.160, 0.171]
    for i in range(num_sims):
        if i < 10:
            if args.mode == "interaction":
                if args.simulation == "springs":
                    sim.interaction_strength = 0.1
                elif args.simulation == "charged":
                    sim.interaction_strength = 1

            elif args.mode == "parameter":
                if args.simulation == "springs":
                    sim.interaction_strength = interaction_list[i]
                elif args.simulation == "charged":
                    sim.interaction_strength = interaction_list[i] * 10

            print(i, "-th simulation interaction: ", sim.interaction_strength)
        else:
            input("We are generating 10 simulations - num_sims are more than 10")

        t = time.time()
        if args.edge_sample_freq == 1:
            # create new interaction graph every simulations
            loc, vel, edges = sim.sample_trajectory(T=length,
                                                    sample_freq=sample_freq)
        else:
            if i % args.edge_sample_freq == 0:
                # create new interaction graph
                loc, vel, edges = sim.sample_trajectory(T=length,
                                                        sample_freq=sample_freq)
            else:
                # re-use previouse interation graph
                loc, vel, edges = sim.sample_trajectory(T=length,
                                                        sample_freq=sample_freq,
                                                        predefined_edges=edges)

        if i % 1 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t), "Interaction Strenght: ", sim.interaction_strength)
        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all

print("Generating {} training simulations".format(args.num_train))
loc_train, vel_train, edges_train = generate_dataset(args.num_train,
                                                     args.length,
                                                     args.sample_freq)

if args.mode == "interaction":
    path_mode = "evolving_interaction/"
    if args.simulation == "springs":
        path_interaction = "_interaction0.1_90k/"
    elif args.simulation == "charged":
        path_interaction = "_interaction1.0_90k/"
elif args.mode == "parameter":
    path_mode = "evolving_parameter/"
    path_interaction = "_interaction_var/"

# NOTE: there is no separate validation or test datasets in online setting
np.save(path_mode + args.simulation + str(args.n_balls) + path_interaction + 'loc_train' + suffix + '.npy', loc_train)
np.save(path_mode + args.simulation + str(args.n_balls) + path_interaction + 'vel_train' + suffix + '.npy', vel_train)
np.save(path_mode + args.simulation + str(args.n_balls) + path_interaction + 'edges_train' + suffix + '.npy', edges_train)