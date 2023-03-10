import numpy as np
import os
import pickle
from scipy.io import loadmat
import matplotlib.pyplot as plt

usePeriodicNetwork = False # True
useRealTraj = True
constantVel = np.array([0., 0.])

# network parameters
tau = 10 # time constant (ms)
if useRealTraj:
    alpha = 50.
else:
    alpha = 0.10315 # parameter used in the original paper
    
n = 128
ncells = 128 * 128
a = 1
lmd = 13
beta = 3 / (lmd ** 2)
gamma = 1.1 * beta # 1.05 * beta in the original paper

spikeThreshold = 0.1

dt = 0.5 # integration time step (ms)
simdur = 100 * 1e3
stabilisationTime = 100 # no-velocity time for pattern to form (ms)
tind = 0
t = 0

s = np.random.random((1, ncells))

watchCell = int(ncells/2) - int(n/2)
nSpatialBins = 60
minx, maxx = -0.9, 0.9
miny, maxy = -0.9, 0.9
occupancy = np.zeros((nSpatialBins, nSpatialBins))
spikes = np.zeros((nSpatialBins, nSpatialBins))
spikeCoords = []

# create 2 x ncells matrix of the 2d cell preferred direction vector (radians)
dirs = np.array([[0, np.pi/2],
                 [np.pi, 3*np.pi/2]])
dirs = np.tile(dirs, [int(n/2), int(n/2)]).reshape(1, -1)
dirVecs = np.concatenate([np.cos(dirs), np.sin(dirs)], axis=0)

# create 2 x ncells matrix of 2d cell positions on the neural sheet
# x = np.linspace(-int(n/2), int(n/2), n)
x = np.arange(n) - (n-1) / 2
X, Y = np.meshgrid(x, x)
x = np.concatenate([X.reshape(1, -1), Y.reshape(1, -1)], axis=0)
cellDists = np.sqrt(np.sum(np.square(x), axis=0))
cellSpacing = Y[1, 0] - Y[0, 0]
ell = 2 * cellSpacing

wSparseThreshold = -1e-6

identifier = ('periodic' if usePeriodicNetwork else 'aperiodic') + '_' + f'ell_{ell}'
if os.path.exists(f'logs/burak_fiete_2009_W_{identifier}.pkl'):
    with open(f'logs/burak_fiete_2009_W_{identifier}.pkl', 'rb') as fi:
        W = pickle.load(fi)
    print('loaded pre-generated W.')
else:
    print('generating new W')
    W = np.zeros((ncells, ncells))
    for i in range(ncells):
        if (i+1) % int(ncells/10) == 0:
            print(f'generating weight matrix, {int((i+1)/ncells*100)}% done')
        if usePeriodicNetwork:
            squaredShiftLengths = np.zeros((9, ncells))
            shifts = np.tile(x[:, [i]], [1, ncells]) - x - ell * dirVecs
            squaredShiftLengths[0, :] = np.sum(np.square(shifts), axis=0)
            shifts = np.tile(x[:, [i]], [1, ncells]) - x - n * np.concatenate([np.ones((1, ncells)), np.zeros((1, ncells))], axis=0) - ell * dirVecs
            squaredShiftLengths[1, :] = np.sum(np.square(shifts), axis=0)
            shifts = np.tile(x[:, [i]], [1, ncells]) - x + n * np.concatenate([np.ones((1, ncells)), np.zeros((1, ncells))], axis=0) - ell * dirVecs
            squaredShiftLengths[2, :] = np.sum(np.square(shifts), axis=0)
            shifts = np.tile(x[:, [i]], [1, ncells]) - x - n * np.concatenate([np.zeros((1, ncells)), np.ones((1, ncells))], axis=0) - ell * dirVecs
            squaredShiftLengths[3, :] = np.sum(np.square(shifts), axis=0)
            shifts = np.tile(x[:, [i]], [1, ncells]) - x + n * np.concatenate([np.zeros((1, ncells)), np.ones((1, ncells))], axis=0) - ell * dirVecs
            squaredShiftLengths[4, :] = np.sum(np.square(shifts), axis=0)
            shifts = np.tile(x[:, [i]], [1, ncells]) - x + n * np.concatenate([np.ones((1, ncells)), np.ones((1, ncells))], axis=0) - ell * dirVecs
            squaredShiftLengths[5, :] = np.sum(np.square(shifts), axis=0)
            shifts = np.tile(x[:, [i]], [1, ncells]) - x + n * np.concatenate([-1 * np.ones((1, ncells)), np.ones((1, ncells))], axis=0) - ell * dirVecs
            squaredShiftLengths[6, :] = np.sum(np.square(shifts), axis=0)
            shifts = np.tile(x[:, [i]], [1, ncells]) - x + n * np.concatenate([np.ones((1, ncells)), -1 * np.ones((1, ncells))], axis=0) - ell * dirVecs
            squaredShiftLengths[7, :] = np.sum(np.square(shifts), axis=0)
            shifts = np.tile(x[:, [i]], [1, ncells]) - x + n * np.concatenate([-1 * np.ones((1, ncells)), -1 * np.ones((1, ncells))], axis=0) - ell * dirVecs
            squaredShiftLengths[8, :] = np.sum(np.square(shifts), axis=0)
            
            squaredShiftLengths = np.min(squaredShiftLengths, axis=0)
        
        else:
            shifts = np.tile(x[:, [i]], [1, ncells]) - x - ell*dirVecs
            squaredShiftLengths = np.sum(np.square(shifts), axis=0)
        
        temp = a * np.exp(-gamma * squaredShiftLengths) - np.exp(-beta * squaredShiftLengths)
        
        temp[temp > wSparseThreshold] = 0
        
        W[i, :] = temp
    
    with open(f'logs/burak_fiete_2009_W_{identifier}.pkl', 'wb') as fi:
        pickle.dump(W, fi)

# define envelope function
if usePeriodicNetwork:
    A = np.ones_like(cellDists)
else:
    R = n / 2
    a0 = n / 32
    dr = n / 2 # diameter of non-tapered region
    A = np.exp(-a0 * ((cellDists-R + dr)/dr)**2)
    A[cellDists < (R-dr)] = 1

pos = loadmat('data/HaftingTraj_centimeters_seconds.mat')['pos']
pos[2, :] = pos[2, :] * 1000 # s to ms

pos = np.concatenate([np.interp(np.arange(0, pos[2, -1]+1), pos[2, :], pos[0, :]).reshape(1, -1), 
                      np.interp(np.arange(0, pos[2, -1]+1), pos[2, :], pos[1, :]).reshape(1, -1), 
                      np.interp(np.arange(0, pos[2, -1]+1), pos[2, :], pos[2, :]).reshape(1, -1)], 
                     axis=0)
pos[:2] = pos[:2] / 100 # cm to m
vels = np.concatenate([(pos[0, 1:]-pos[0, :-1]).reshape(1, -1), 
                       (pos[1, 1:]-pos[1, :-1]).reshape(1, -1)], 
                      axis=0) / dt

x, y = pos[0, 0], pos[1, 0]

speed = np.zeros((1, int(simdur / dt)))
curDir = np.zeros((1, int(simdur / dt)))
vhist = np.zeros((1, int(simdur / dt)))
fhist = np.zeros((1, int(simdur / dt)))
spikeCoords = []

while t < simdur:
    tind += 1
    t = dt * tind
    
    if t < stabilisationTime:
        v = np.array([0., 0.])
    else:
        v = vels[:, tind]
    
    curDir[0, tind] = np.arctan2(v[1], v[0])
    speed[0, tind] = np.sqrt(v[0]**2 + v[1]**2)
    
    # feedforard input
    B = A * (1 + alpha*(dirVecs.T.dot(v)).T)
    
    sInputs = (W.dot(s.T)).T + B
    sInputs = sInputs * (sInputs > 0)
    
    s = s + dt * (sInputs - s) / tau
    
    if useRealTraj:
        if s[0, watchCell] > spikeThreshold:
            spikeCoords.append([pos[0, tind], pos[1, tind]])
            xindex = int((pos[0, tind] - minx) / (maxx-minx) * nSpatialBins)
            yindex = int((pos[1, tind] - miny) / (maxy-miny) * nSpatialBins)
            occupancy[yindex, xindex] += dt
            spikes[yindex, xindex] += s[0, watchCell]
    
    if tind % 40 == 0:
        pass