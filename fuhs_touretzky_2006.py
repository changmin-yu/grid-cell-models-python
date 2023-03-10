import numpy as np
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import pickle
# from scipy.interpolate import interp1d

def Phi(x):
    # weight matrix definition (note that this is a crude approximation rather than computed with 
    # actual numerical integration)
    return 0.5 * np.exp(-0.25 * np.abs(x)) * np.cos(2 * np.pi * np.abs(x) / 7)

useRealTraj = True
constantVel = .1 * np.array([.5, 0]) / 1e3

tau = 10 # grid cell synapse time constant (ms)

n = 162
ncells = n * n # total number of cell sin the network
omega = 0.67 # wave function frequency (see table 1 of Fuhs and Touretzky, 2006)
alphaSym = 1 # changed from 0.5 in the original paper;
alphaAsym = -1.5
psi1 = 7/4 # cutoff value, changed from 2.55 in the original paper
sigmaGamma = 13.375 # weight fadeaway annulus
beta = 0.75 # asymmetric offset, changed from 1.5 in the original paper
velGain = 1000 # gain on velocity input

# simulation parameters
dt = 1
simdur = 120e3 # total simulation time (ms)
stabilisationTime = 200 # no-velocity time for pattern to form (ms)
tind = 0
t = 0
x, y = 0, 0

speed = np.zeros((1, int(simdur / dt)))
curDir = np.zeros((1, int(simdur / dt)))
vhist = np.zeros((1, int(simdur / dt)))
fhist = np.zeros((1, int(simdur / dt)))

watchCell = 1970
nSpatialBins = 60
minx, maxx = -0.9, 0.9
miny, maxy = -0.9, 0.9
occupancy = np.zeros((nSpatialBins, nSpatialBins))
spikes = np.zeros((nSpatialBins, nSpatialBins))

spikeCoords = []

V = np.random.random((1, ncells))
f = np.zeros((1, ncells))

# create 2 x ncells matrix of the 2d cell preferred direction vector (radians)
dirs = np.array([[0, np.pi/2],
                 [np.pi, 3*np.pi/2]])
dirs = np.tile(dirs, [int(n/2), int(n/2)]).reshape(1, -1)
dirVecs = np.concatenate([np.cos(dirs), np.sin(dirs)], axis=0)

# create 2 x ncells matrix of 2d cell positions on the neural sheet
x = np.linspace(-int(n/2), int(n/2), n)
X, Y = np.meshgrid(x, x)
x = np.concatenate([X.reshape(1, -1), Y.reshape(1, -1)], axis=0)
ell = beta / omega
cellDists = np.sqrt(np.sum(np.square(x), axis=0))
circleMask = cellDists < int(n / 2)
gamma = np.exp(-(cellDists / sigmaGamma)**4) # \gamma_{j} from eq.6

# always generate new W from scratch

if not os.path.exists('logs/fuhs_touretzky_W.pkl'):

    W = np.zeros((ncells, ncells))
    tic = time.time()
    for i in range(ncells):
        if (i+1) % int(0.2 * ncells) == 0:
            print(f'generating weight matrix {int((i+1)/ncells * 100)} done, time taken: {time.time()-tic:.2f}')
        
        if cellDists[i] > int(n/2):
            W[i, :] = 0.
            continue
        
        shifts = np.tile(x[:, [i]], [1, ncells]) - x - ell * dirVecs
        asymDists = np.sqrt(np.sum(np.square(shifts), axis=0))
        wasym = alphaAsym * np.exp(-(omega * asymDists / 1.5)**2)/2 # this seems off from equation 4, especially the cutoff!
        
        shifts = np.tile(x[:, [i]], [1, ncells]) - x
        symDists = np.sqrt(np.sum(np.square(shifts), axis=0))
        wsym = alphaSym * Phi(omega * symDists)
        
        W[i, :] = circleMask * gamma * (wsym + wasym)
        
    with open('logs/fuhs_touretzky_W.pkl', 'wb') as fi:
        pickle.dump(W, fi)

else:
    with open('logs/fuhs_touretzky_W.pkl', 'rb') as fi:
        W = pickle.load(fi)
    
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

print('Simulation starting...')
while t < simdur:
    tind += 1
    t = dt * tind
    
    v = vels[:, tind]
    v = velGain * v
    curDir[0, tind] = np.arctan2(v[1], v[0])
    speed[0, tind] = np.sqrt(v[0]**2 + v[1]**2)
    
    x = x + v[0] * dt
    y = y + v[1] * dt
    
    velInput = 1/2 + 2 * speed[0, tind] * np.exp(-np.sin((curDir[0, tind]-dirs)/2)**2/(0.245**2) - 1/4)
    
    V = V + dt * ((W.dot(f.T)).T + velInput - V + 0.2 * np.random.randn(1, ncells)) / tau
    
    vhist[0, tind] = V[0, int(ncells/2)-int(n/2)]
    
    f = circleMask * np.sqrt(V * (V>0))
    fhist[0, tind] = f[0, int(ncells/2)-int(n/2)]
    
    if useRealTraj:
        if f[0, watchCell] > 0:
            spikeCoords = spikeCoords.append([pos[0, tind], pos[1, tind]])
            xindex = int((pos[0, tind] - minx) / (maxx-minx) * nSpatialBins)
            yindex = int((pos[1, tind] - miny) / (maxy-miny) * nSpatialBins)
            occupancy[yindex, xindex] += dt
            spikes[yindex, xindex] += f[0, watchCell]
    
    if tind % 30 == 0:
        # plt.imshow(spikes / (occupancy+1e-8))
        pass