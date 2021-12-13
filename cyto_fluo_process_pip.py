# -*- coding: utf-8 -*-

"""
 @auther: Pan M. CHU
"""

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import pandas as pd
import numpy as np  # Or any other
# […]
from typing import List, Tuple

import utils as utl

# Own modules

tubename_list = List[str]


def sor_min(unsort: tubename_list) -> Tuple[str, tubename_list]:
    order = lambda name: (ord(name.split('-')[-1][0]) - ord('A'))*12 + int(name.split('-')[-1][1:])
    order_list = [order(na) for na in unsort]
    last = unsort[0]
    last_order = order_list[0]
    index = 0
    for i, na in enumerate(order_list):
        if na < last_order:
            last_order = na
            last = unsort[i]
            index = i
    del unsort[index]
    return last, unsort


def iter_sample(unsort):
    while unsort:
        na, unsort = sor_min(unsort)
        yield na


# %%
dir = r'F:\New folder\Exp_20211207_1'

tube_dic = utl.parallel_process_fsc(dir)
tube_name = list(tube_dic.keys())
tube_name.sort()
# name_iter = iter_sample(tube_name)
# tube_name = [na for na in name_iter]
data_list = []
for name in tube_name:
    data_frame = tube_dic[name].statistic
    fluo_row = data_frame.loc['Green-H']
    data_list.append(fluo_row.to_list() + [name])

columns_na = list(fluo_row.index) + ['Tube_name']

all_statistic = pd.DataFrame(data=data_list, columns=columns_na)

all_statistic.to_csv(os.path.join(dir, 'statistic_all.csv'))
