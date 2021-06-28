# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:33:38 2018

@author: FuLab
"""
#%%
import os
import numpy as np
import pandas as pd
import api
from scipy import stats
import matplotlib.pyplot as plt
#%%
data_path= r'E:\Fu_lab_Data\cytometer data\20200313'
file_lis = [f.name[:-4] for f in os.scandir(path = data_path) if f.is_file()]       # capture file name
flowdatadic = {x: api.FCSParser(path = data_path + '\\'+ str(x)+'.fcs').dataframe for x in file_lis} # open all file and put into a dictionary

da1 = flowdatadic['1-2.fcs']
x = da1['FSC-A']
y = da1['SSC-A']
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
#perform a kernel density estimate on the data:
X,Y = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]
positions = np.vstack([X.ravel(),Y.ravel()])
values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
ax.plot(x,y, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()