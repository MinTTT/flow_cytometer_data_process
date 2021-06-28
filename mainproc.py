# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:57:37 2018

@author: FuLab
"""
import numpy as np
import matplotlib.pyplot as plt
import api
import pandas as pd

def mycv(me):
    return np.mean(me)/np.std(me)*100

da1 = api.FCSParser(path = r'E:\Pan\毕设\cytometry\20171124\Exp_20171124_1\cl2-g1.fcs',
                    channel_naming='$PnN')
da = da1.dataframe

plt.figure(figsize=(4,4))
daFSCH = da["FSC-H"]
daSSCH = da["SSC-H"]
daFL3A = da["FL3-A"]
cvf = mycv(daFL3A)
cvfc = mycv(daFL3A/daFSCH)
print('%s%%, %s%% ' % (str(cvf), str(cvfc) ))
plt.scatter(daFL3A/daFSCH, daFL3A, alpha = '0.1')
plt.xscale('log')
plt.yscale('log')
plt.show()


