# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import os
import sys
# […]
import matplotlib.collections
import matplotlib.pyplot as plt

import utils as utl
from kde_scatter_plot import kde_plot
from sciplot import whitegrid

whitegrid()
# Libs
import pandas as pd
import numpy as np  # Or any other
# […]
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Or any other
from scipy.stats import gaussian_kde, binned_statistic, linregress
import matplotlib.animation as animation
import sciplot as splt

splt.whitegrid()
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

cmp = cm.get_cmap('coolwarm')


# Own modules


def update_canv(index, path: matplotlib.collections.Collection):
    tubename = tube_name[index]
    tube_info = info_df[info_df['label'] == tubename]
    fmt_label = '''#%s Time: %.2f h $\lambda$: %.2f $\mathrm{h^{-1}}$''' % \
                (str(index), tube_info['Time (h)'], tube_info['growth rate'])
    channel_data = tube_dic[tubename].masked_data
    data = channel_data[[channel_dict['x'], channel_dict['y']]].values
    fsc_h = channel_data['FSC-H'].values
    data_xy = data / fsc_h.reshape(-1, 1) * 1000
    density = gaussian_kde(np.log(data_xy[:500, :].T))
    colors = density(np.log(data_xy[:, :].T))
    colr_norm = cm.colors.Normalize(vmin=colors.min(), vmax=colors.max())
    normed_colors = colr_norm(colors)
    # color = cmp(density(data_xy.T))
    path.set_offsets(data_xy[:, :])
    path.set_array(normed_colors)
    time_text.set_text(fmt_label)
    return path


# %%

dir = r'\\FH_Group_Server\homes\jingwen zhu\exp_data\toggle_NCM\202106\20210630_FCS\L3_G'
channel_dict = dict(x='ECD-H', y='FITC-H')
info_df = pd.read_excel(r'\\FH_Group_Server\homes\jingwen zhu\exp_data\toggle_NCM\202106\20210630_FCS\L3_G\L3_G_data.xlsx',
                        usecols=range(5))



tube_dic = utl.parallel_process_fsc(dir)
tube_name = list(tube_dic.keys())
tube_name.sort(key=lambda name: int(name.split('_')[-1]))
#%%


fig1, ax = plt.subplots(1, 1)
path = ax.scatter([], [], c=[], cmap='coolwarm', s=5)
time_text = ax.text(1e4*0.001, 5e3*0.7, s='')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 1e4)
ax.set_ylim(1, 5e3)
ax.grid(False)
ani = animation.FuncAnimation(fig1, update_canv, frames=range(len(tube_name)), fargs=(path,),
                              interval=500)

fig1.show()


writervideo = animation.FFMpegWriter(fps=30)

ani.save(os.path.join(dir, 'animation.mp4'), writer=writervideo)

#%%
for index, tna in enumerate(tube_name):
    fig1, ax = plt.subplots(1, 1)
    tubename = tna
    tube_info = info_df[info_df['label'] == tubename]
    fmt_label = '''#%s Time: %.2f h $\lambda$: %.2f $\mathrm{h^{-1}}$''' % \
                (str(index), tube_info['Time (h)'], tube_info['growth rate'])
    channel_data = tube_dic[tna].masked_data
    data = channel_data[[channel_dict['x'], channel_dict['y']]].values


    fsc_h = channel_data['FSC-H'].values
    data_xy = data / fsc_h.reshape(-1, 1) * 1000
    density = gaussian_kde(np.log(data_xy[:500, :].T))
    colors = density(np.log(data_xy[:, :].T))
    colr_norm = cm.colors.Normalize(vmin=colors.min(), vmax=colors.max())
    normed_colors = colr_norm(colors)
    # color = cmp(density(data_xy.T))
    ax.scatter(data_xy[:, 0], data_xy[:, 1], c=normed_colors.data, cmap='coolwarm', s=8)
    # path = kde_plot(data_xy, s=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 1e4)
    ax.set_ylim(1, 5e3)
    ax.text(1e4*0.001, 5e3*0.7, s=fmt_label)
    ax.grid(False)
    fig1.savefig(os.path.join(r'\\FH_Group_Server\homes\jingwen zhu\exp_data\toggle_NCM\202106\20210630_FCS\L3_G\figs', f'{index}.png'))

