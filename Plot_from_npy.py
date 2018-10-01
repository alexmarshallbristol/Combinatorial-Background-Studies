import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


single_background = np.load('single_muon_track_info.npy')

pair_background = np.load('pair_muon_track_info.npy')

# single_muon_track_info = np.append(single_muon_track_info, [[weight, nmeas, rchi2, P, Px, Py, Pz]], axis=0)
# pair_muon_track_info = np.append(pair_muon_track_info, [[weight, doca, zv, dist, pair_mom]],axis=0)

print(np.shape(single_background), np.shape(pair_background))


#need to plot errors too

plt.hist(single_background[:,3],weights=single_background[:,0],normed=True,bins=25)
plt.savefig('plots/single_mom',bbox_inches='tight')
plt.close('all')

plt.hist(pair_background[:,4],weights=pair_background[:,0],normed=True,bins=25)
plt.savefig('plots/pair_mom',bbox_inches='tight')
plt.close('all')

plt.hist(pair_background[:,1],weights=pair_background[:,0],normed=True,bins=25)
plt.savefig('plots/doca',bbox_inches='tight')
plt.close('all')

plt.hist(pair_background[:,2],weights=pair_background[:,0],normed=True,bins=25,range=[-10000,10000])
plt.savefig('plots/z_vertex',bbox_inches='tight')
plt.close('all')

plt.hist(pair_background[:,3],weights=pair_background[:,0],normed=True,bins=25)
plt.yscale('log')
plt.savefig('plots/IP',bbox_inches='tight')
plt.close('all')
