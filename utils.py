import os
import numpy as np
import api
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
from joblib import Parallel, delayed
import fcswrite as fcwr
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def processing(tn, dic, thre=-25):
    da1 = dic[tn]
    data_comp = da1[['FSC-H', 'FSC-Width', 'SSC-H']]
    data_comp.insert(len(data_comp.columns), 'SSC-W', da1['SSC-A'] / da1['SSC-H'])
    data_comp = data_comp.dropna(axis=0, how='any')
    gmm = mixture.GaussianMixture(n_components=2, reg_covar=10e-4).fit(data_comp)
    # posteriors = gmm.predict_proba(data_comp)
    scores = gmm.score_samples(data_comp)
    mask = scores > thre
    data_trim = da1.iloc[data_comp[mask].index]
    gfp_mean_h = data_trim['FITC-H'].mean() / data_trim['FSC-H'].mean()
    gfp_mean_a = data_trim['FITC-A'].mean() / data_trim['FSC-A'].mean()
    gfp_ind_h = data_trim['FITC-H'] / data_trim['FSC-H']
    gfp_ind_a = data_trim['FITC-A'] / data_trim['FSC-A']
    gfp_ind_h_mean = np.mean(gfp_ind_h)
    gfp_ind_a_mean = np.mean(gfp_ind_a)
    # data_trim['FLU-H'] = gfp_ind_h
    # data_trim['FLU-A'] = gfp_ind_a
    data_trim.insert(len(data_trim.columns), 'FLU-H', gfp_ind_h)
    data_trim.insert(len(data_trim.columns), 'FLU-A', gfp_ind_a)
    return data_trim, [gfp_mean_h, gfp_mean_a, gfp_ind_h_mean, gfp_ind_a_mean]


class CytoTube():
    def __init__(self, name, dirc):
        self._tube_name = name
        self._tube_dir = dirc
        self._raw_data = api.FCSParser(path=os.path.join(self._tube_dir, self._tube_name+'.fcs')).dataframe
        # dirc + '\\' + name + '.fcs'
        self._lasso_mask = None
        self._score = None
        self._mask = None
        self._gmm = None
        self._threshold = None
        self.threshold_bottom_line = None
        self.threshold_range = None
        self.statistic = pd.DataFrame(data=None)  # type: pd.DataFrame
        self.masked_data = None

    def gate(self, threshold=None):
        """
        threshold: float, if set None, this method will set the model threshold automatically.
        """
        # print('Gating tube %s automatically via gaussian model.' % self._tube_name)
        self._raw_data.insert(len(self._raw_data.columns), 'FLU-H', self._raw_data['FITC-H'] / self._raw_data['FSC-H'] * 1000)
        self._raw_data.insert(len(self._raw_data.columns), 'FLU-A', self._raw_data['FITC-A'] / self._raw_data['FSC-A'] * 1000)
        df_for_model = self._raw_data[['FSC-H', 'FSC-Width', 'SSC-H']]
        df_for_model.insert(len(df_for_model.columns),
                            'SSC-W', self._raw_data['SSC-A'] / self._raw_data['SSC-H'])
        self._gmm = mixture.GaussianMixture(n_components=2, reg_covar=10e-4).fit(df_for_model)
        self._score = self._gmm.score_samples(df_for_model)
        scorted_score = np.sort(self._score)
        self.threshold_range = [scorted_score[0], scorted_score[-1]]

        if threshold is None:
            threshold = scorted_score[int(len(scorted_score) * 0.85) - 1]
            self.threshold_bottom_line = scorted_score[int(len(scorted_score) * 0.05) - 1]
        self._threshold = threshold
        self._mask = self._score > threshold
        print('Threshold of %s is %f.\n' % (self._tube_name, threshold))

        self.masked_data = self._raw_data[self._mask]
        self.masked_data.index = np.arange(len(self.masked_data))
        self.set_statistic()

    def tune_gate(self, threshold: float):
        self._threshold = threshold
        # print('Thre: %f' % threshold)
        self._mask = self._score > threshold
        self.masked_data = self._raw_data[self._mask]
        self.masked_data.index = np.arange(len(self.masked_data))
        self.set_statistic()

    def set_statistic(self, sample_num=3):
        std = self.masked_data.std(axis=0)
        mean = self.masked_data.mean(axis=0)
        cv = std / mean
        scv = cv ** 2
        median = pd.Series(data=np.median(self.masked_data, axis=0), index=mean.index)
        rcv = np.median(np.abs(self.masked_data.values - median.T.values),
                        axis=0) * 1.4826 / median.values.flatten()
        rcv = pd.Series(data=rcv, index=mean.index)
        data_dic = dict(std=std, mean=mean, cv=cv, scv=scv, median=median, rcv=rcv)
        events_index = np.arange(len(self.masked_data))
        events_class = np.zeros(len(self.masked_data))
        np.random.shuffle(events_index)
        split_index = list(split(events_index, sample_num))
        group_na_mean = []
        group_na_medium = []

        for i in range(sample_num):
            events_class[split_index[i]] = i
            data_dic[f'group_{i}_mean'] = self.masked_data.iloc[split_index[i], :].mean(axis=0)
            data_dic[f'group_{i}_medium'] = pd.Series(data=np.median(self.masked_data.iloc[split_index[i], :], axis=0),
                                                      index=mean.index)
            group_na_medium.append(f'group_{i}_medium')
            group_na_mean.append(f'group_{i}_mean')

        # self.masked_data['group'] = events_class
        self.masked_data.insert(len(self.masked_data.columns), 'group', events_class)
        self.statistic = pd.DataFrame(data=data_dic)
        self.statistic['group_mean_std'] = self.statistic[group_na_mean].std(axis=1)
        self.statistic['group_mean_average'] = self.statistic[group_na_mean].mean(axis=1)
        self.statistic['group_medium_std'] = self.statistic[group_na_medium].std(axis=1)
        self.statistic['group_medium_average'] = self.statistic[group_na_medium].mean(axis=1)


    # def add_channel(self):
def parallel_process_fsc(data_dir):
    def process_func(obj):
        obj.gate()

    file_lis = [f.name[:-4] for f in os.scandir(path=data_dir) if
                (f.is_file() and f.name[-3:] == 'fcs')]  # capture file name

    tube_data_dic = {x: CytoTube(x, data_dir) for x in file_lis}

    _ = Parallel(n_jobs=-1, verbose=True, require='sharedmem')(delayed(process_func)(tube_data_dic[tube_na]) for tube_na in file_lis)

    return tube_data_dic




# %%


if __name__ == '__main__':
    # %%
    data_path = r'F:\20210611_607psc101 mvenus\exprt_202110611'
    tube_data_dic = parallel_process_fsc(data_path)
    file_lis = list(tube_data_dic.keys())
    file_lis.sort()

    #%%
    import sys
    sys.path.append(r'D:\python_code\toggle_mini')
    import sciplot as splt
    splt.whitegrid()

    fig1, ax1 = plt.subplots(1, 1, figsize=(20, 10))
    for tube_na in file_lis:
        conc0 = tube_data_dic[tube_na].masked_data['FLU-H']
        tube_data_dic[tube_na].masked_data.to_csv(os.path.join(data_path, f'{tube_na}.csv'))
        ax1.hist(conc0, range=(10, conc0.max()),
                 density=True, label=tube_na, bins=1000, histtype='step', lw=5)

    ax1.set_xscale('log')
    ax1.set_xlim(15, 1.5e3)
    ax1.grid(False)
    ax1.legend()
    ax1.set_xlabel('Green fluorescent intensity (a.u.)')
    ax1.set_ylabel('probability')
    ax1.tick_params(direction='in')
    ax1.tick_params(direction='in')

    fig1.show()
    fig1.savefig(os.path.join(data_path, 'hcy_2.svg'))



