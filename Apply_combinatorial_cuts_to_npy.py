import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

single_background = np.load('single_muon_track_info_not1overweight.npy')

pair_background = np.load('pair_muon_track_info_not1overweight.npy')

# single_muon_track_info = np.append(single_muon_track_info, [[weight, nmeas, rchi2, P, Px, Py, Pz]], axis=0)
# pair_muon_track_info = np.append(pair_muon_track_info, [[weight, doca, zv, dist, pair_mom, fid, mom_i_mag, mom_j_mag]],axis=0)

print(np.shape(single_background), np.shape(pair_background))


total_rate_pre_cut = np.sum(pair_background[:,0])
print('total rate:', total_rate_pre_cut)

#doca
to_delete = np.where(pair_background[:,1] > 1)
pair_background = np.delete(pair_background, to_delete, axis=0)

#ip
to_delete = np.where(pair_background[:,3] > 250)
pair_background = np.delete(pair_background, to_delete, axis=0)

# to_delete = np.where(pair_background[:,4] < 1)

# pair_background = np.delete(pair_background, to_delete, axis=0)

#individual track mom
to_delete = np.where(pair_background[:,6] < 1)
pair_background = np.delete(pair_background, to_delete, axis=0)
to_delete = np.where(pair_background[:,7] < 1)
pair_background = np.delete(pair_background, to_delete, axis=0)


#fiducial
to_delete = np.where(pair_background[:,5] == 0)
pair_background = np.delete(pair_background, to_delete, axis=0)


print(np.shape(pair_background), np.sum(pair_background[:,0]), np.sum(pair_background[:,0])/total_rate_pre_cut)

# need to get errors - (sqrt(sum(weights**2)) for that that got through) same for that that didnt get through - combine in quadrature for ratio error 



#signal
pair_signal = np.load('pair_muon_track_info_signal.npy')


total_rate_pre_cut_signal = np.sum(pair_signal[:,0])
print('total rate signal:', total_rate_pre_cut_signal)

#doca
to_delete = np.where(pair_signal[:,1] > 1)
pair_signal = np.delete(pair_signal, to_delete, axis=0)

#ip
to_delete = np.where(pair_signal[:,3] > 250)
pair_signal = np.delete(pair_signal, to_delete, axis=0)

# to_delete = np.where(pair_background[:,4] < 1)

# pair_background = np.delete(pair_background, to_delete, axis=0)

#individual track mom
to_delete = np.where(pair_signal[:,6] < 1)
pair_signal = np.delete(pair_signal, to_delete, axis=0)
to_delete = np.where(pair_signal[:,7] < 1)
pair_signal = np.delete(pair_signal, to_delete, axis=0)


#fiducial
to_delete = np.where(pair_signal[:,5] == 0)
pair_signal = np.delete(pair_signal, to_delete, axis=0)


print(np.shape(pair_signal), np.sum(pair_signal[:,0]), np.sum(pair_signal[:,0])/total_rate_pre_cut_signal)






