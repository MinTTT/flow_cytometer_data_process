import os
import numpy as np
# import api
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
from joblib import Parallel, delayed
import flowio
import FlowCal
from typing import Optional, Dict, Union


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def parse_fcs(data_ps):

    tube = flowio.FlowData(data_ps, ignore_offset_error=True)
    tube_raw_data = np.reshape(tube.events, (-1, tube.channel_count))
    # channel_names = [tube.channels[channel_key]['PnS'] for channel_key in list(tube.channels.keys())]
    channel_names = []
    for channel_key in list(tube.channels.keys()):
        try:
            channel_names.append(tube.channels[channel_key]['PnS'])
        except KeyError:
            channel_names.append(tube.channels[channel_key]['PnN'])
    tube_df = pd.DataFrame(data=tube_raw_data, columns=channel_names)

    return tube_df


class CytoTube():
    """
    Create a fcs tube objection

    Parameters
    ---------
    name: string, default is None
        tube name, if have, please specific.

    ps: string
        the .fcs file path.
    """

    def __init__(self, ps: Optional[str] = None, name: Optional[str] = None):
        self._tube_name = name
        self._tube_ps = ps
        if self._tube_ps is None:
            raise FileNotFoundError('Please check the file path.')

        self._raw_data = parse_fcs(ps)  # read file
        self.channels = self._raw_data.columns.to_list()
        self._lasso_mask = None
        self._score = None
        self._mask = None
        self._gmm = None
        self._threshold = None
        self.threshold_bottom_line = None
        self.threshold_range = None
        self.statistic = pd.DataFrame(data=None)  # type: pd.DataFrame
        self.masked_data = None

    def new_stat(self):
        if ('FITC-H' in self.channels) and ('FSC-H' in self.channels):
            self._raw_data.insert(len(self._raw_data.columns), 'Green-H',
                                  self._raw_data['FITC-H'] / self._raw_data['FSC-H'] * 1000.)
        if ('FITC-A' in self.channels) and ('FSC-A' in self.channels):
            self._raw_data.insert(len(self._raw_data.columns), 'Green-A',
                                  self._raw_data['FITC-A'] / self._raw_data['FSC-A'] * 1000.)
        if ('ECD-H' in self.channels) and ('FSC-H' in self.channels):
            self._raw_data.insert(len(self._raw_data.columns), 'Red-H',
                                  self._raw_data['ECD-H'] / self._raw_data['FSC-H'] * 1000.)
        return None

    def gate(self, threshold=None):

        """
        threshold: float, if set None, this method will set the model threshold automatically.
        """
        try:
            df_for_model = self._raw_data[['FSC-H', 'FSC-Width', 'SSC-H']]
        except KeyError:
            df_for_model = self._raw_data[['FSC-H', 'SSC-H']]
            df_for_model.insert(len(df_for_model.columns),
                                'FSC-W', self._raw_data['FSC-A'] / self._raw_data['FSC-H'])
        df_for_model.insert(len(df_for_model.columns),
                            'SSC-W', self._raw_data['SSC-A'] / self._raw_data['SSC-H'])
        self._gmm = mixture.GaussianMixture(n_components=2, reg_covar=10e-4).fit(df_for_model)
        self._score = self._gmm.score_samples(df_for_model)
        sorted_score = np.sort(self._score)
        self.threshold_range = [sorted_score[0], sorted_score[-1]]

        if threshold is None:
            threshold = sorted_score[int(len(sorted_score) * 0.85) - 1]
            self.threshold_bottom_line = sorted_score[int(len(sorted_score) * 0.05) - 1]
        self._threshold = threshold
        self._mask = self._score > threshold
        print('Threshold of %s is %f.\n' % (self._tube_name, threshold))

        self.masked_data = self._raw_data[self._mask]
        self.masked_data.index = np.arange(len(self.masked_data))

    def tune_gate(self, threshold: float):
        self._threshold = threshold
        # print('Thre: %f' % threshold)
        self._mask = self._score > threshold
        self.masked_data = self._raw_data[self._mask]
        self.masked_data.index = np.arange(len(self.masked_data))
        self.set_statistic()

    def set_statistic(self, sample_num=1):
        """
        Calculate the statistics

        Parameters
        --------
        sample_num: int
            resample number, divide
        """
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


def parallel_process_fsc(data_path: str, file_list=None):
    def process_func(obj: CytoTube):
        obj.new_stat()
        obj.gate()
        obj.set_statistic()

    all_fcs_list = [f.name for f in os.scandir(path=data_path)
                    if (f.is_file() and f.name[-3:] == 'fcs')]  # capture file name
    if file_list is not None:
        tube_names = [file_na.strip('.fcs') for file_na in file_list if file_na in all_fcs_list]
    else:
        tube_names = [name.strip('.fcs') for name in all_fcs_list]

    tube_data_dic = {tube_names[i]: CytoTube(os.path.join(data_path, f'{file_name}.fcs'), name=tube_names[i])
                     for i, file_name in enumerate(tube_names)}

    _ = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(process_func)(tube_data_dic[tube_name]) for tube_name in tube_names)

    return tube_data_dic


# %%
if __name__ == '__main__':
    # %%
    from kde_scatter_plot import hist_sample, kde_plot
    data_path = r'\\FH_Group_Server\homes\panchu\experiment_data\cytometry\20220213_cymR_SR_library\Exp_20220213_1'
    file_names = ['CymR_SR_F8.fcs', 'CymR_SR_H12.fcs', 'CymR_SR_F1.fcs', 'CymR_SR_H1.fcs',
                  'CymR_SR_C10-2.fcs', 'CymR_SR_F2.fcs', 'CymR_SR_G2.fcs',
                  'CymR_SR_G5.fcs', 'CymR_SR_H2.fcs', 'CymR_SR_H4.fcs']
    # file_names = ['L3_pIPTG_10.fcs', '02-Well-B3.fcs']
    tube_data_dic = parallel_process_fsc(data_path, file_names)
    file_lis = list(tube_data_dic.keys())
    file_lis.sort()

    list_tube_name = list(tube_data_dic.keys())

    fig = hist_sample({key: tube_data_dic[key].masked_data for key in list_tube_name}, stat_key='Green-H',
                      bins_num=400, xlim=(5, 2000))

    axes = fig.get_axes()

    # for ax in axes:
    #     ax.set_xlim(100, 2e4)

    fig.show()
    fig.savefig(os.path.join(data_path, 'statistic_cell_counts.svg'))

    #
    # fig2, ax = plt.subplots(1, 1)
    # ax.contour(tube_data_dic['02-Well-B3']._raw_data['ECD-H'])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    #
    # fig2.show()
    # %%
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
