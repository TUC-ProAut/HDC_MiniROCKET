# Kenny Schlegel, Peer Neubert, Peter Protzel
#
# HDC-MiniROCKET : Explicit Time Encoding in Time Series Classification with Hyperdimensional Computing
# Copyright (C) 2022 Chair of Automation Technology / TU Chemnitz

# file to create the plot of graded similarity in timesteps

import numpy as np
import matplotlib.pyplot as plt
from models.HDC_MINIROCKET import HDC_MINIROCKET_model
from config import *
from sklearn.metrics.pairwise import cosine_similarity

D = 10000

config = Config_orig()
config.HDC_dim = D
HDC_model = HDC_MINIROCKET_model(config)

t_init = (np.random.rand(D)>0.5)*2-1
n_steps = 100
n_scales = config.scales.shape[0]
sim_mat = np.zeros((n_scales,n_steps))
legend = []
t = np.arange(1,101)

# scale range
scales = config.scales

for s_idx in range(len(scales)):
    scale = scales[s_idx]
    HDC_model.create_pose_matrix(n_steps,scale)
    t_0 = HDC_model.poses[0,:]
    sim_mat[s_idx,:] = cosine_similarity(np.expand_dims(t_0,0),HDC_model.poses)
    legend.append('s = ' + str(np.round(scale,2)))
plt.plot(t,np.transpose(sim_mat,(1,0)))
plt.grid()
plt.xlabel('timestamp difference as percent of total series length')
plt.ylabel('cosine similarity of vector encodings')
plt.legend(legend)
plt.savefig('graded_sim_timestamps.png')
plt.show()