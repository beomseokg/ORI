import numpy as np

spring_edges = np.load('evolving_interaction/springs10_interaction0.1_90k/edges_train_springs10.npy')
spring_loc = np.load('evolving_interaction/springs10_interaction0.1_90k/loc_train_springs10.npy')
spring_vel = np.load("evolving_interaction/springs10_interaction0.1_90k/vel_train_springs10.npy")

charge_edges = np.load('evolving_interaction/charged10_interaction1.0_90k/edges_train_charged10.npy')
charge_loc = np.load('evolving_interaction/charged10_interaction1.0_90k/loc_train_charged10.npy')
charge_vel = np.load("evolving_interaction/charged10_interaction1.0_90k/vel_train_charged10.npy")

new_edges = np.zeros_like(charge_edges)
new_loc = np.zeros_like(charge_loc)
new_vel = np.zeros_like(charge_vel)

# springs -> charged -> springs -> charged -> ...
dynamics = ['s', 'c', 'c', 's', 'c', 's', 'c', 's', 's', 'c']
for i in range(10):
    if dynamics[i] == 's':
        new_edges[i] = spring_edges[i]
        new_loc[i] = spring_loc[i]
        new_vel[i] = spring_vel[i]
    elif dynamics[i] == 'c':
        new_edges[i] = charge_edges[i]
        new_loc[i] = charge_loc[i]
        new_vel[i] = charge_vel[i]

np.save('evolving_dynamics/mixed/edges_train_mix10.npy', new_edges)
np.save('evolving_dynamics/mixed/loc_train_mix10.npy', new_loc)
np.save('evolving_dynamics/mixed/vel_train_mix10.npy', new_vel)