import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


def main(date, id):
    kbr_df = dd.read_csv(urlpath='..//input/KBR1A_' + date + '_' + id + '_04.txt', sep='\s+',
                          engine='c', skiprows=235, header=None,
                          storage_options=dict(auto_mkdir=False), names=['rcv_time_intg',
                          'rcv_time_frac', 'id', 'prn_id', 'ant_id', 'prod_flag', 'qualflg',
                          'k_phase', 'ka_phase', 'k_snr', 'ka_snr'],
                          dtype={'k_phase': np.longdouble, 'ka_phase': np.longdouble}, encoding='gb2312')
    k_phase = kbr_df.k_phase.compute().to_numpy()
    ka_phase = kbr_df.ka_phase.compute().to_numpy()

    k_mask = np.array(np.where(np.abs(np.diff(k_phase)) > 1000000.0)) + 1; k_mask = k_mask[0]
    ka_mask = np.array(np.where(np.abs(np.diff(ka_phase)) > 1000000.0)) + 1; ka_mask = ka_mask[0]

    sign = 0.0
    if id == 'C':
        sign = -1.0
    else:
        sign = 1.0

    for _, index in enumerate(k_mask):
        k_phase[index:] = np.add(k_phase[index:], 100000000.0 * sign, dtype=np.longdouble)
    for _, index in enumerate(ka_mask):
        ka_phase[index:] = np.add(ka_phase[index:], 100000000.0 * sign, dtype=np.longdouble)

    np.set_printoptions(threshold=sys.maxsize)
    phase = np.c_[k_phase, ka_phase]
    phase = pd.DataFrame(phase)
    phase.to_csv('..//output//phase_wrap_' + date + '_' + id + '.txt', index=False)


if __name__ == '__main__':
    main('2019-01-01', 'D')
