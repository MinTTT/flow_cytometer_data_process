#%%
import os
import numpy as np
import api
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd

#%%
data_path = r'E:\Fu_lab_Data\cytometer data\20200327-ratio\Exp_20200327__MOPS_EZ_GLY'
file_lis = [f.name[:-4] for f in os.scandir(path=data_path) if f.is_file() and f.name[-3:] == 'fcs']       # capture file name
flowdatadic = {x: api.FCSParser(path=data_path+'\\'+str(x)+'.fcs').dataframe
               for x in file_lis}# open all file and put into a dictionary
#%%
da1 = flowdatadic['2-2']
# da1 = np.log(da1)
da1 = da1.dropna(axis=0, how='any')
x = da1['FSC-A']
y = da1['SSC-A']
# xmin = x.min()
# xmax = x.max()
# ymin = y.min()
# ymax = y.max()
plt.figure(figsize=(8, 8))
c = plt.hexbin(x, y, gridsize=500, cmap='jet', bins='log')
# plt.axis([xmin, xmax, ymin, ymax])
plt.colorbar(c)
plt.xscale('log')
plt.yscale('log')
plt.show()




#%% multivaribale gaussian
# thehold = 0.5
# da1 = flowdatadic['#151']
# data_comp = da1[['FSC-H', 'FSC-Width', 'SSC-H']]
# data_comp['SSC-W'] = da1['SSC-A'] / da1['SSC-H']
# data_comp = np.log(data_comp).dropna(axis=0, how='any')
# gmm = mixture.GaussianMixture(n_components=1).fit(data_comp)
# posteriors = gmm.predict_proba(data_comp)
# scores = gmm.score_samples(data_comp)
# mask = scores > thehold
#
# x = data_comp['FSC-H']
# y = data_comp['SSC-H']
# plt.figure(figsize=(8, 8))
# c = plt.hexbin(np.log(x), np.log(y), gridsize=250, cmap='jet', bins='log')
# plt.scatter(np.log(x[mask]), np.log(y[mask]))
# # plt.axis([xmin, xmax, ymin, ymax])
# plt.colorbar(c)
# plt.show()


#%%
thehold = -25
da1 = flowdatadic['#151']
data_comp = da1[['FSC-H', 'FSC-Width', 'SSC-H']]
data_comp['SSC-W'] = da1['SSC-A'] / da1['SSC-H']
data_comp = data_comp.dropna(axis=0, how='any')
gmm = mixture.GaussianMixture(n_components=2).fit(data_comp)
posteriors = gmm.predict_proba(data_comp)
scores = gmm.score_samples(data_comp)
mask = scores > thehold






x = data_comp['FSC-H']
y = data_comp['SSC-H']
plt.figure(figsize=(8, 8))
c = plt.hexbin(np.log(x), np.log(y), gridsize=250, cmap='jet', bins='log')
plt.scatter(np.log(x[mask]), np.log(y[mask]))
# plt.axis([xmin, xmax, ymin, ymax])
plt.colorbar(c)
plt.show()


#%%
data_trim = da1.iloc[data_comp[mask].index]
gfp_mean = data_trim['FITC-H'].mean()/data_trim['FSC-H'].mean()
gfp_ind = data_trim['FITC-H']/data_trim['FSC-H']
gfp_ind_a = data_trim['FITC-A']/data_trim['FSC-A']
gfp_ind_mean = np.mean(gfp_ind)
gfp_a = data_trim['FITC-A']
gfp_h = data_trim['FITC-H']
fsc_a = data_trim['FSC-A']
fsc_h = data_trim['FSC-H']
fig2, ax = plt.subplots(1, 4)
ax[0].hist(gfp_ind, bins=10000)
ax[1].hist(gfp_a, bins=10000)
ax[2].hist(gfp_h, bins=10000)
ax[3].scatter(fsc_a, gfp_ind)
# ax[1].set_xlim(-1000, 800)
# ax.set_xlim(0, 1)
plt.show()


covs = np.corrcoef([gfp_ind, gfp_ind_a, gfp_a, fsc_a, fsc_h])
print(covs)
#%%
import utils
from tqdm import tqdm
import os
import numpy as np
import api
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
import seaborn as sns

data_path = r'E:\Fu_lab_Data\cytometer data\old\Exp_20200110_1_fLU'
file_lis = [f.name[:-4] for f in os.scandir(path=data_path) if (f.is_file() and f.name[-3:] == 'fcs')]       # capture file name
flowdatadic = {x: api.FCSParser(path=data_path+'\\'+str(x)+'.fcs').dataframe
               for x in file_lis}# open all file and put into a dictionary

data_trim_dic = dict()
data_list = list()
for nam in file_lis:
    print('prccessing %s' % nam)
    pf, datl = utils.processing(nam, flowdatadic, thre=-28)
    data_trim_dic[nam] = pf
    data_list.append(datl)
#%%
def get_coeffc(mat):
    '''
    FLU-H FLU-A FSC-H FSC-A
    '''
    return mat[:2, -2:].reshape(1, -1)

covs = list()
for nam in file_lis:
    df = data_trim_dic[nam]
    cov = np.corrcoef([df['FLU-H'], df['FLU-A'], df['FSC-H'], df['FSC-A']])
    covs.append(get_coeffc(cov))

covs = np.vstack(covs).reshape(-1, 1)
tag = ['GFPH-FSCH', 'GFPH-FSCA', 'GFPA-FSCH', 'GFPA-FSCA'] * len(file_lis)


#%%
covs_df = pd.DataFrame(data=covs, columns=['coefs'])
covs_df['tag'] = tag

fig, ax = plt.subplots(1, 1)
sns.boxplot(x='tag', y='coefs', data=covs_df, ax=ax)
fig.show()


#%%
da1.to_csv(r'./sample.csv')