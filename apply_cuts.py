import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

single_background = np.load('single_muon_track_info_not1overweight.npy')

pair_background = np.load('pair_muon_track_info_not1overweight.npy')

# single_muon_track_info = np.append(single_muon_track_info, [[weight, nmeas, rchi2, P, Px, Py, Pz]], axis=0)
# pair_muon_track_info = np.append(pair_muon_track_info, [[weight, doca, zv, dist, pair_mom]],axis=0)

print(np.shape(single_background), np.shape(pair_background))


total_rate_pre_cut = np.sum(pair_background[:,0])
print('total rate:', total_rate_pre_cut)

to_delete = np.where(pair_background[:,1] > 1)

pair_background = np.delete(pair_background, to_delete, axis=0)

to_delete = np.where(pair_background[:,3] > 250)

pair_background = np.delete(pair_background, to_delete, axis=0)

to_delete = np.where(pair_background[:,4] < 1)

pair_background = np.delete(pair_background, to_delete, axis=0)

#if fiducial


print(np.shape(pair_background), np.sum(pair_background[:,0]), np.sum(pair_background[:,0])/total_rate_pre_cut)














