import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

single_background = np.load('single_muon_track_info_not1overweight.npy')

pair_background = np.load('pair_muon_track_info_not1overweight.npy')

single_signal = np.load('signal/single_muon_track_info_signal.npy')

pair_signal = np.load('signal/pair_muon_track_info_signal.npy')

# print(np.shape(single_signal), np.shape(pair_signal))
# print(np.shape(single_signal), np.shape(pair_signal))

# single_muon_track_info = np.append(single_muon_track_info, [[weight, nmeas, rchi2, P, Px, Py, Pz]], axis=0)
# pair_muon_track_info = np.append(pair_muon_track_info, [[weight, doca, zv, dist, pair_mom]],axis=0)

# print(np.shape(single_background), np.shape(pair_background))

print(np.sum(single_signal[:,0]), np.sum(single_background[:,0]))

print(np.sum(pair_signal[:,0]), np.sum(pair_background[:,0]))

# quit()

#need to plot errors too


# def get_errors(signal_array, signal_weights, background_array, background_weights, min_value, max_value, bins):

# 	# signal_weights = signal_weights * (np.sum(background_weights)/np.sum(signal_weights))

# 	signal_hist = np.histogram(signal_array,weights=signal_weights, bins=bins,range=[min_value, max_value],normed=True)
# 	plot_sig = np.empty((0,3))
# 	for i in range(0, np.shape(signal_hist[0])[0]):
# 		plot_sig = np.append(plot_sig, [[(signal_hist[1][i]+signal_hist[1][i+1])/2,signal_hist[0][i],math.sqrt(signal_hist[0][i])]], axis=0)
# 		# use weighted errors

# 	background_hist = np.histogram(background_array,weights=background_weights, bins=bins,range=[min_value, max_value],normed=True)
# 	plot_bkg = np.empty((0,3))
# 	for i in range(0, np.shape(background_hist[0])[0]):
# 		plot_bkg = np.append(plot_bkg, [[(background_hist[1][i]+background_hist[1][i+1])/2,background_hist[0][i],math.sqrt(background_hist[0][i])]], axis=0)
# 	return plot_sig, plot_bkg


def get_errors(array, weights, start_value, end_value, bins):
	'''
	Return normalised plot with errors.
	'''
	bin_width = (end_value - start_value)/bins
	x = np.empty(bins)
	y = np.empty(bins)
	yerr = np.empty(bins)
	for i in range(0, bins):
		x[i] = i*bin_width + bin_width/2 + start_value
		to_delete = np.where(array > x[i] + bin_width/2)
		array_bin = np.delete(array,to_delete)
		weights_bin = np.delete(weights,to_delete)

		to_delete = np.where(array_bin < x[i] - bin_width/2)
		array_bin = np.delete(array_bin,to_delete)
		weights_bin = np.delete(weights_bin,to_delete)

		y[i] = np.sum(weights_bin)

		weights_bin_sq = weights_bin*weights_bin

		yerr[i] = math.sqrt(np.sum(weights_bin_sq))

	sum_y = np.sum(y)
	# print(sum_y)
	y = y/sum_y
	yerr = yerr/sum_y

	return x, y, yerr




pair_mom_sig_x, pair_mom_sig_y, pair_mom_sig_yerr = get_errors(pair_signal[:,4], pair_signal[:,0], 0, 250, 35)
pair_mom_bkg_x, pair_mom_bkg_y, pair_mom_bkg_yerr = get_errors(pair_background[:,4], pair_background[:,0], 0, 250, 35)
plt.errorbar(pair_mom_sig_x, pair_mom_sig_y, yerr=pair_mom_sig_yerr, color='#FF5959',label='Signal', capsize=4, marker='s',markersize=3,linewidth=1,elinewidth=2)
plt.errorbar(pair_mom_bkg_x, pair_mom_bkg_y, yerr=pair_mom_bkg_yerr, color='#4772FF',label='Background', capsize=4, marker='s',markersize=3,linewidth=1,elinewidth=2)
plt.legend(loc='upper right')
plt.ylim(0,1)
plt.savefig('plots/pair_mom_er',bbox_inches='tight')
plt.close('all')

# print(' ')
single_mom_sig_x, single_mom_sig_y, single_mom_sig_yerr = get_errors(single_signal[:,3], single_signal[:,0], 0, 250, 35)
single_mom_bkg_x, single_mom_bkg_y, single_mom_bkg_yerr = get_errors(single_background[:,3], single_background[:,0], 0, 250, 35)
plt.errorbar(single_mom_sig_x, single_mom_sig_y, yerr=single_mom_sig_yerr, color='#FF5959',label='Signal', capsize=4, marker='s',markersize=3,linewidth=1,elinewidth=2)
plt.errorbar(single_mom_bkg_x, single_mom_bkg_y, yerr=single_mom_bkg_yerr, color='#4772FF',label='Background', capsize=4, marker='s',markersize=3,linewidth=1,elinewidth=2)
plt.legend(loc='upper right')
plt.ylim(0,1)
plt.savefig('plots/single_mom_er',bbox_inches='tight')
plt.close('all')
# print(' ')

pair_doca_sig_x, pair_doca_sig_y, pair_doca_sig_yerr = get_errors(pair_signal[:,1], pair_signal[:,0], 0, 800, 35)
pair_doca_bkg_x, pair_doca_bkg_y, pair_doca_bkg_yerr = get_errors(pair_background[:,1], pair_background[:,0], 0, 800, 35)
plt.errorbar(pair_doca_sig_x, pair_doca_sig_y, yerr=pair_doca_sig_yerr, color='#FF5959',label='Signal', capsize=4, marker='s',markersize=3,linewidth=1,elinewidth=2)
plt.errorbar(pair_doca_bkg_x, pair_doca_bkg_y, yerr=pair_doca_bkg_yerr, color='#4772FF',label='Background', capsize=4, marker='s',markersize=3,linewidth=1,elinewidth=2)
plt.legend(loc='upper right')
plt.ylim(0,1)
plt.savefig('plots/pair_doca_er',bbox_inches='tight')
plt.close('all')

pair_ip_sig_x, pair_ip_sig_y, pair_ip_sig_yerr = get_errors(pair_signal[:,3], pair_signal[:,0], 0, 1000, 35)
pair_ip_bkg_x, pair_ip_bkg_y, pair_ip_bkg_yerr = get_errors(pair_background[:,3], pair_background[:,0], 0, 1000, 35)
plt.errorbar(pair_ip_sig_x, pair_ip_sig_y, yerr=pair_ip_sig_yerr, color='#FF5959',label='Signal', capsize=4, marker='s',markersize=3,linewidth=1,elinewidth=2)
plt.errorbar(pair_ip_bkg_x, pair_ip_bkg_y, yerr=pair_ip_bkg_yerr, color='#4772FF',label='Background', capsize=4, marker='s',markersize=3,linewidth=1,elinewidth=2)
plt.legend(loc='upper right')
plt.ylim(0,1)
plt.savefig('plots/pair_ip_er',bbox_inches='tight')
plt.close('all')

pair_vz_sig_x, pair_vz_sig_y, pair_vz_sig_yerr = get_errors(pair_signal[:,2], pair_signal[:,0], -10000, 10000, 35)
pair_vz_bkg_x, pair_vz_bkg_y, pair_vz_bkg_yerr = get_errors(pair_background[:,2], pair_background[:,0], -10000, 10000, 35)
plt.errorbar(pair_vz_sig_x, pair_vz_sig_y, yerr=pair_vz_sig_yerr, color='#FF5959',label='Signal', capsize=4, marker='s',markersize=3,linewidth=1,elinewidth=2)
plt.errorbar(pair_vz_bkg_x, pair_vz_bkg_y, yerr=pair_vz_bkg_yerr, color='#4772FF',label='Background', capsize=4, marker='s',markersize=3,linewidth=1,elinewidth=2)
plt.legend(loc='upper right')
plt.ylim(0,1)
plt.savefig('plots/pair_vz_er',bbox_inches='tight')
plt.close('all')









