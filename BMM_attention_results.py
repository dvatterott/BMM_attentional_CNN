import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pickle
import numpy as np


performance_avg1_att = pickle.load( open( "./Performance/minst_att_nonoise_performance.p", "rb" ) )
performance_avg2_att = pickle.load( open( "./Performance/minst_att1_nonoise_performance.p", "rb" ) )
performance_avg3_att = pickle.load( open( "./Performance/minst_att2_nonoise_performance.p", "rb" ) )

performance_ones1_att = pickle.load( open( "./Performance/minst_onesatt_nonoise_performance.p", "rb" ) )
performance_ones2_att = pickle.load( open( "./Performance/minst_onesatt1_nonoise_performance.p", "rb" ) )
performance_ones3_att = pickle.load( open( "./Performance/minst_onesatt2_nonoise_performance.p", "rb" ) )

#performance_avg1_att = pickle.load( open( "./Performance/minst_att_noise_performance.p", "rb" ) )
#performance_avg2_att = pickle.load( open( "./Performance/minst_att1_noise_performance.p", "rb" ) )
#performance_avg3_att = pickle.load( open( "./Performance/minst_att2_noise_performance.p", "rb" ) )

#performance_ones1_att = pickle.load( open( "./Performance/minst_onesatt_noise_performance.p", "rb" ) )
#performance_ones2_att = pickle.load( open( "./Performance/minst_onesatt1_noise_performance.p", "rb" ) )
#performance_ones3_att = pickle.load( open( "./Performance/minst_onesatt2_noise_performance.p", "rb" ) )

performance_ones_att = np.mean([[performance_ones1_att],[performance_ones2_att],[performance_ones2_att]],axis=0)[0]
performance_avg_att = np.mean([[performance_avg1_att],[performance_avg2_att],[performance_avg2_att]],axis=0)[0]

ste_ones_att = np.std([[performance_ones1_att],[performance_ones2_att],[performance_ones2_att]],axis=0)[0]/(3.**(1./2))
ste_avg_att = np.std([[performance_avg1_att],[performance_avg2_att],[performance_avg2_att]],axis=0)[0]/(3.**(1./2))

size_ratio = 10
f, (ax, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[size_ratio, 1]})

ax.set_ylim(50, 100)  # outliers only
ax2.set_ylim(0, 12)  # most of the data

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
ax2.set_yticks([0,10])

x = np.arange(len(performance_avg_att))+1

ax.plot(x,performance_avg_att*100,'-',label='With Spatial Attention')
ax.plot(x,performance_ones_att*100,'-',label='Without Spatial Attention')
ax.fill_between(x, performance_ones_att*100+ste_ones_att*100, performance_ones_att*100-ste_ones_att*100, facecolor='green',alpha=0.25)
ax.fill_between(x, performance_avg_att*100+ste_avg_att*100, performance_avg_att*100-ste_avg_att*100, facecolor='blue',alpha=0.25)
ax2.plot(x,performance_avg_att*100,'-',label='With Spatial Attention')
ax2.plot(x,performance_ones_att*100,'-',label='Without Spatial Attention')
ax2.legend(loc=4,frameon=False)
ax2.set_xlabel('Training Epoch #')
ax.set_ylabel('Validation Accuracy')
ax2.set_xlim([0,13])

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes,)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d*size_ratio, 1 + d*size_ratio), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d*size_ratio, 1 + d*size_ratio), **kwargs)  # bottom-right diagonal

fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
yticks = mpl.ticker.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)
ax2.yaxis.set_major_formatter(yticks)

plt.subplots_adjust(hspace=0.05)

plt.savefig('./images/model_performance_nonoise.png', dpi=300,bbox_inches='tight')

