import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

array = np.load('pair_muon_track_info_not1overweight_no_cuts.npy')

print(np.shape(array))

#delete where 
# pair_muon_track_info = np.append(pair_muon_track_info, [[weight, doca, zv, dist, pair_mom, fid, mom_i_mag, mom_j_mag,single_muon_track_info[i][1],single_muon_track_info[j][1],single_muon_track_info[i][2],single_muon_track_info[j][2]]],axis=0)


delete_this = np.where(single_muon_track_info[:,7] == 0)
single_muon_track_info = np.delete(single_muon_track_info, delete_this,axis=0)
fittedTracks = np.delete(fittedTracks, delete_this,axis=0)
list_of_fitted_states = np.delete(list_of_fitted_states, delete_this,axis=0)


total_weight = np.sum(array[:,0])

weights_sq = array[:,0]*array[:,0]

error = math.sqrt(np.sum(weights_sq))

print(total_weight, error, 'as %',(error/total_weight)*100)