from typing import Optional
import numpy as np
import os
import pickle
from scipy.io import loadmat
import matplotlib.pyplot as plt

from tqdm import trange


class BurakFiete2009:
    def __init__(
        self, 
        tau: float = 10, # time constant
        alpha: float = 50., # multiplicative scalar factor coupling rat velocity to network dynamics
        a: float = 1., # scaling factor for the center-surround weight distribution
        side_len: int = 128, # number of discretised bins on each side of the square enclosure (neural sheet)
        lmd: int = 13, # approximate periodicity of the formed lattice in the neural sheet
        gamma_scaling: float = 1.1, # scaling factor between gamma and beta, 1.05 is used in the original paper
        spike_threshold: float = 0.1, # integrate-and-fire threshold
        dt: float = 0.5, # integration time step
        duration: float = 1e5, # simulation duration
        stabilisation_time: float = 100, # no-velocity time for pattern formation
        periodic: bool = False, # using periodic boundary conditions in the neural sheet
        use_real_traj: bool = True, # using real trajectrory
        num_spatial_bins: int = 60, # number of spatial bins
        w_sparse_threshold: float = -1e-6, 
    ):
        self.tau = tau
        self.alpha = alpha
        self.a = a
        
        self.side_len = side_len
        self.num_neurons = side_len * side_len
        
        self.lmd = lmd
        self.beta = 3 / (lmd ** 2)
        self.gamma = gamma_scaling * self.beta
        
        self.spike_threshold = spike_threshold
        self.dt = dt
        self.duration = duration
        self.stabilisation_time = stabilisation_time
        
        self.periodic = periodic
        self.use_real_traj = use_real_traj
        
        self.watch_cell = self.num_neurons // 2 - self.side_len // 2
        
        self.num_spatial_bins = num_spatial_bins
        self.x_min = -0.9
        self.x_max = 0.9
        self.y_min = -0.9
        self.y_max = 0.9
        
        directions = np.array([
            [0, np.pi / 2], 
            [np.pi, 3 * np.pi / 2],
        ])
        self.directions = np.tile(directions, [self.side_len // 2, self.side_len // 2]).reshape(1, -1)
        self.direction_vecs = np.concatenate([np.cos(self.directions), np.sin(self.directions)], axis=0)
        
        self.construct_neural_sheet()
        
        self.w_sparse_threshold = w_sparse_threshold # sparsity constraint in constructing weight matrix
        self.W = self.construct_weight_matrix()
        self.A = self.construct_envelope()
    
    def construct_neural_sheet(self):
        x = np.arange(self.side_len) - (self.side_len - 1) / 2
        X, Y = np.meshgrid(x, x)
        self.neural_sheet = np.concatenate([X.reshape(1, -1), Y.reshape(1, -1)], axis=0)
        self.cell_distance = np.sqrt(np.sum(np.square (self.neural_sheet), axis=0))
        self.cell_spacing = Y[1, 0] - Y[0, 0] # length of field shift in recurrent connections
        self.ell = 2 * self.cell_spacing # offset of center of inhibitory output
        
    def construct_weight_matrix(self):
        identifier = "burak_fiete_2009_W_" + ("periodic" if self.periodic else "aperiodic") + \
            f"N_{self.num_neurons}_ell_{self.ell}"
        
        if os.path.exists(f"logs/{identifier}.pkl"):
            with open(f"logs/{identifier}.pkl", "rb") as f:
                W = pickle.load(f)
            f.close()

            print("Cached W matrix loaded!")
        else:
            print("Generating weight matrix")
            
            W = np.zeros((self.num_neurons, self.num_neurons))
            with trange(self.num_neurons, dynamic_ncols=True) as pbar:
                for i in pbar:
                    if self.periodic:
                        squared_shift_length = np.zeros((9, self.num_neurons))
                        shifts = np.tile(self.neural_sheet[:, [i]], [1, self.num_neurons]) - self.neural_sheet - self.ell * self.direction_vecs
                        squared_shift_length[0, :] = np.sum(np.square(shifts), axis=0)
                        shifts = np.tile(self.neural_sheet[:, [i]], [1, self.num_neurons]) - self.neural_sheet - self.ell * self.direction_vecs - self.side_len * np.concatenate([np.ones((1, self.num_neurons)), np.zeros((1, self.num_neurons))], axis=0)
                        squared_shift_length[1, :] = np.sum(np.square(shifts), axis=0)
                        shifts = np.tile(self.neural_sheet[:, [i]], [1, self.num_neurons]) - self.neural_sheet - self.ell * self.direction_vecs + self.side_len * np.concatenate([np.ones((1, self.num_neurons)), np.zeros((1, self.num_neurons))], axis=0)
                        squared_shift_length[2, :] = np.sum(np.square(shifts), axis=0)
                        shifts = np.tile(self.neural_sheet[:, [i]], [1, self.num_neurons]) - self.neural_sheet - self.ell * self.direction_vecs - self.side_len * np.concatenate([np.zeros((1, self.num_neurons)), np.ones((1, self.num_neurons))], axis=0)
                        squared_shift_length[3, :] = np.sum(np.square(shifts), axis=0)
                        shifts = np.tile(self.neural_sheet[:, [i]], [1, self.num_neurons]) - self.neural_sheet - self.ell * self.direction_vecs + self.side_len * np.concatenate([np.zeros((1, self.num_neurons)), np.ones((1, self.num_neurons))], axis=0)
                        squared_shift_length[4, :] = np.sum(np.square(shifts), axis=0)
                        shifts = np.tile(self.neural_sheet[:, [i]], [1, self.num_neurons]) - self.neural_sheet - self.ell * self.direction_vecs + self.side_len * np.concatenate([np.ones((1, self.num_neurons)), np.ones((1, self.num_neurons))], axis=0)
                        squared_shift_length[5, :] = np.sum(np.square(shifts), axis=0)
                        shifts = np.tile(self.neural_sheet[:, [i]], [1, self.num_neurons]) - self.neural_sheet - self.ell * self.direction_vecs + self.side_len * np.concatenate([-1 * np.ones((1, self.num_neurons)), np.ones((1, self.num_neurons))], axis=0)
                        squared_shift_length[6, :] = np.sum(np.square(shifts), axis=0)
                        shifts = np.tile(self.neural_sheet[:, [i]], [1, self.num_neurons]) - self.neural_sheet - self.ell * self.direction_vecs + self.side_len * np.concatenate([np.ones((1, self.num_neurons)), -1 * np.ones((1, self.num_neurons))], axis=0)
                        squared_shift_length[7, :] = np.sum(np.square(shifts), axis=0)
                        shifts = np.tile(self.neural_sheet[:, [i]], [1, self.num_neurons]) - self.neural_sheet - self.ell * self.direction_vecs - self.side_len * np.concatenate([np.ones((1, self.num_neurons)), np.ones((1, self.num_neurons))], axis=0)
                        squared_shift_length[8, :] = np.sum(np.square(shifts), axis=0)
                        
                        squared_shift_length = np.min(squared_shift_length, axis=0)
                    else:
                        shifts = np.tile(self.neural_sheet[:, [i]], [1, self.num_neurons]) - self.neural_sheet - self.ell * self.direction_vecs
                        squared_shift_length = np.sum(np.square(shifts), axis=0)

                    w_temp = self.a * np.exp(-self.gamma * squared_shift_length) - np.exp(-self.beta * squared_shift_length)
                    w_temp[w_temp > self.w_sparse_threshold] = 0
                    
                    W[i, :] = w_temp
            
            with open(f"logs/{identifier}.pkl", "wb") as f:
                pickle.dump(W, f)                    
            f.close()
        
        return W
    
    def construct_envelope(self):
        if self.periodic:
            A = np.ones_like(self.cell_distance)
        else:
            R = self.side_len / 2
            a0 = self.side_len / 32
            dr = self.side_len / 2
            A = np.exp(-a0 * np.square((self.cell_distance - R + dr)/dr))
            A[self.cell_distance < dr] = 1.
        
        return A
    
    def load_real_traj(self, logdir: str):
        assert os.path.exists(logdir)
        
        pos = loadmat(logdir)["pos"]
        pos[2, :] = pos[2, :] * 1000 # s -> ms
        
        pos = np.concatenate([
            np.interp(np.arange(0, pos[2, -1] + 1), pos[2, :], pos[0, :]).reshape(1, -1),
            np.interp(np.arange(0, pos[2, -1] + 1), pos[2, :], pos[1, :]).reshape(1, -1),
            np.interp(np.arange(0, pos[2, -1] + 1), pos[2, :], pos[2, :]).reshape(1, -1),
        ], axis=0)
        
        pos[:2] /= 100 # cm -> m
        
        velocity = np.concatenate([
            (pos[0, 1:] - pos[0, :-1]).reshape(1, -1), 
            (pos[1, 1:] - pos[1, :-1]).reshape(1, -1), 
        ], axis=0) / self.dt
        
        return pos, velocity
    
    def simulation(self, logdir: Optional[str] = None):
        if self.use_real_traj:
            assert logdir is not None
            
            pos, velocity = self.load_real_traj(logdir)
            
            simulation_length = int(self.duration / self.dt)
            
            speed = np.zeros((simulation_length, ))
            curr_direction = np.zeros((simulation_length, ))
            occupancy = np.zeros((self.num_spatial_bins, self.num_spatial_bins))
            spikes = np.zeros((self.num_spatial_bins, self.num_spatial_bins))
            spike_coordinates = []
            
            t = 0
            s = np.random.random((self.num_neurons, ))
            with trange(simulation_length, dynamic_ncols=True) as pbar:
                for i in pbar:
                    t = i * self.dt

                    if t < self.stabilisation_time:
                        v = np.zeros((2, ))
                    else:
                        v = velocity[:, i]
                    
                    curr_direction[i] = np.arctan2(v[1], v[0])
                    speed[i] = np.sqrt(v[0] ** 2 + v[1] ** 2)
                    
                    # feedforward input
                    B = self.A * (1 + self.alpha * np.dot(v, self.direction_vecs))
                    
                    s_inputs = self.W.dot(s) + B
                    s_inputs = s_inputs * (s_inputs > 0)
                    
                    s = s + self.dt * (s_inputs - s) / self.tau
                    
                    if s[self.watch_cell] > self.spike_threshold:
                        spike_coordinates.append([pos[0, i], pos[1, i]])
                        x_ind = int((pos[0, i] - self.x_min) / (self.x_max - self.x_min) * self.num_spatial_bins)
                        y_ind = int((pos[1, i] - self.y_min) / (self.y_max - self.y_min) * self.num_spatial_bins) 
                        
                        occupancy[y_ind, x_ind] += self.dt
                        spikes[y_ind, x_ind] += s[self.watch_cell]
                    
                    pbar.set_description(f"t = {t}ms / {self.duration}ms")
        else:
            raise NotImplementedError

        return spike_coordinates, occupancy, spikes


if __name__=="__main__":
    traj_logdir = "data/HaftingTraj_centimeters_seconds.mat"
    bf_can = BurakFiete2009()
    
    # takes about 1 hour to run the simulation over 20mins recording!
    spike_coordinates, occupancy, spikes = bf_can.simulation(traj_logdir)
    