import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

# column = ['work_time', 'engine_speed', 'pump_speed', 'pumping_pressure', 'oil_temperature',
#           'flow_stall', 'distribution_pressure', 'displacement_current', 'low_voltage_switch', 'high_voltage_switch',
#           'overpressure_signal', 'positive_pump', 'negative_pump', 'equipment_type', 'id']

column = ['c' + str(i) for i in range(15)]

raw_path = 'raw_data/'
data_path = 'data/'
deal_path = 'deal/'
feat_path = 'feat/'
model_path = 'model/'


def init_data():
    """ read all csv to pickle """

    data_train = raw_path + 'data_train/'
    file_map = {name: i for (i, name) in enumerate(os.listdir(data_train))}

    df_list = []
    for file in tqdm(list(file_map.keys())):
        df = pd.read_csv(data_train + file)
        df['file_id'] = pd.Series([file_map[file]] * len(df), index=df.index)
        df_list.append(df)

    all_df = pd.concat(df_list, axis=0, ignore_index=True)
    all_df.to_pickle(data_path + 'train.pickle')

def deal_one_csv(file):
    data = pd.read_csv(file)
    rename_dic = {data.columns[i]: column[i] for i in range(15)}
    data = data.rename(columns=rename_dic)

    agg_func = {
        'c0': ['mean'],
        'c1': ['mean'],
        'c2': ['mean'],
        'c3': ['mean'],
        'c4': ['mean'],
        'c5': ['mean'],
        'c6': ['mean'],
        'c7': ['mean'],
        'c8': ['sum'],
        'c9': ['sum'],
        'c10': ['sum'],
        'c11': ['sum'],
        'c12': ['sum'],
        # 'c12': ['sum'],
    }

if __name__ == '__main__':
    init_data()
