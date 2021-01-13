#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, kaiserord, firwin, filtfilt
import glob
import dask.dataframe as dd
import platform
from scipy.stats import pearsonr, norm


def brush_1_0(xy):
    arr_1 = xy[xy[:, 1] == 1]


def iono_corr():
    # cutoff frequency (Hz)
    fcut = 0.1

    # read in data
    iono_corr = np.loadtxt(fname='..//output//IONO1A_2019-01-01_Y_03.txt',
                           dtype=np.longdouble,
                           skiprows=1)
    dd_range = np.loadtxt('..//..//..//gracefo_dataset//gracefo_1A_2019-01-01_RL04.ascii.noLRI//DDR1A_2019-01-01_Y_04.txt',
                          dtype=np.longdouble,
                          skiprows=1,
                          max_rows=863998)
    dowr_k = np.loadtxt(fname='..//..//..//gracefo_dataset//gracefo_1A_2019-01-01_RL04.ascii.noLRI//DOWR1A_2019-01-01_Y_04.txt',
                        usecols=1,
                        dtype=np.longdouble)
    dowr_ka = np.loadtxt(fname='..//..//..//gracefo_dataset//gracefo_1A_2019-01-01_RL04.ascii.noLRI//DOWR1A_2019-01-01_Y_04.txt',
                        usecols=2,
                        dtype=np.longdouble)
    dd_kbr1b = dd.read_csv(urlpath='..//input//KBR1B_2019-01-01_Y_04.txt',
                           engine='c',
                           header=None,
                           sep='\s+',
                           skiprows=162,
                           dtype=np.longdouble,
                           names=[
                               'gps_time', 'biased_range', 'range_rate', 'range_accl',
                               'iono_corr', 'lighttime_err', 'lighttime_rate',
                               'lighttime_accl', 'ant_centr_corr', 'ant_centr_rate',
                               'ant_centr_accl', 'k_a_snr', 'ka_a_snr', 'k_b_snr', 'ka_b_snr', 'qualflg'
                           ])
    shadow = np.loadtxt(fname='..//..//..//gracefo_dataset//gracefo_1A_2019-01-01_RL04.ascii.noLRI//SHA1A_2019-01-01_C_04.txt')
    dd_kbr1a_c = dd.read_csv(urlpath='..//..//..//gracefo_dataset//gracefo_1A_2019-01-01_RL04.ascii.noLRI//KBR1A_2019-01-01_C_04.txt',
                             engine='c',
                             header=None,
                             names=[
                                 'time_intg', 'time_frac', 'gracefo_id',
                                 'prn_id', 'ant_id', 'prod_flag', 'qualflg',
                                 'k_phase', 'ka_phase', 'k_snr', 'ka_snr'
                             ],
                             skiprows=250,
                             sep='\s+',)
    # filter
    iono_kbr1b = dd_kbr1b.iono_corr.compute().to_numpy()
    iono_corr_hf = iono_corr - kaiser(iono_corr, 5., fcut)
    iono_corr_hf = dd_range - kaiser(dd_range, 5., fcut)
    iono_corr_hf = dowr_k - kaiser(dowr_k, 5., fcut)
    iono_corr_hf = dowr_ka - kaiser(dowr_ka, 5., fcut)
    temp = iono_corr_hf[1500: 862500]
    temp = temp[0::10]
    power_hf_0 = np.sum(np.abs(temp[shadow[150: 86250, 1] == 0])) / temp[shadow[150: 86250, 1] == 0].__len__()
    power_hf_1 = np.sum(np.abs(temp[shadow[150: 86250, 1] == 1])) / temp[shadow[150: 86250, 1] == 1].__len__()
    var_hf_0 = np.var(temp[shadow[150: 86250, 1] == 0])
    var_hf_1 = np.var(temp[shadow[150: 86250, 1] == 1])
    print(power_hf_0, power_hf_1)
    print(var_hf_0, var_hf_1)
    freq_corr, psd_corr = welch(iono_corr,
                                10., ('kaiser', 30.),
                                iono_corr.__len__(),
                                scaling='density')
    freq_ddra, psd_ddra = welch(dd_range,
                                10., ('kaiser', 30.),
                                iono_corr.__len__(),
                                scaling='density')
    freq_k1ac, psd_k1ac = welch(dd_range,
                                10., ('kaiser', 30.),
                                dd_kbr1a_c.__len__(),
                                scaling='density')
    brush_1_0(shadow)

    # norm
    mu, std = norm.fit(iono_corr_hf)

    # plt.style.use(['science', 'no-latex', 'high-vis',])

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dd_range - kaiser(dd_range, 5., fcut), linewidth=2)
    ax.set_xlabel(r'$Sampling \,\, point$', fontsize=20)
    ax.set_ylabel(r'$Double \,\, Differenced \,\, Range [m]$',
                  fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=20, loc='best', frameon=False)
    ax.tick_params(labelsize=25, width=2.9)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlabel(r'$Sampling \,\, point$', fontsize=20)
    ax.set_ylabel(r'$Ionosphere \,\, correction (Ka \,\, band) [m]$', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=20, loc='best', frameon=False)
    ax.tick_params(labelsize=25, width=2.9)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.loglog(freq_corr, np.sqrt(psd_corr), linewidth=2, label='Ka band')
    ax.set_xlabel(r'$Frequency [Hz]$', fontsize=20)
    ax.set_ylabel(r'$ASD [m/\sqrt{Hz}]$', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=20, loc='best', frameon=False)
    ax.tick_params(labelsize=25, width=2.9)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.loglog(freq_ddra, np.sqrt(psd_ddra), linewidth=2, label='double differenced range')
    ax.semilogx(np.linspace(0.0001, 5, 10000),
                2.62 *
                np.sqrt(1 + (0.003 / np.linspace(0.0001, 5, 10000)**2)) * 1e-6,
                label='stochastic error requirement')
    ax.set_xlabel(r'$Frequency [Hz]$', fontsize=20)
    ax.set_ylabel(r'$ASD [m/\sqrt{Hz}]$', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=20, loc='best', frameon=False)
    ax.tick_params(labelsize=25, width=2.9)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(np.linspace(0, 86400, iono_corr_hf.__len__()), iono_corr_hf, linewidth=2, label='Ka')
    ax.plot(np.linspace(0, 86400, iono_corr_hf.__len__()), np.zeros(iono_corr_hf.__len__()), linewidth=2, label=u'拟合')
    ax.fill_between(shadow[:, 0] - shadow[0, 0], 0, 1, where=shadow[:, 1] == 1,
                    color='grey', alpha=0.5, transform=ax.get_xaxis_transform())
    ax.set_ylim([-4e-5, 4e-5])
    ax.set_xlim([150, 86250])
    ax.set_xlabel(u"自2019年1月1日00:00:00开始GPS时", fontsize=20)
    ax.set_ylabel(u'Ka波段星间距高频噪声 [m]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=20, loc='best', frameon=False)
    ax.tick_params(labelsize=25, width=2.9)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.hist(iono_corr_hf,
            bins=np.linspace(-0.0001, 0.0001, 100),
            label='Ka band')
    ax.plot(np.linspace(-0.0001, 0.0001, 100),
            norm.pdf(np.linspace(-0.0001, 0.0001, 100).astype(float), mu.astype(float), std.astype(float)), 'k', linewidth=2)
    ax.set_xlabel(r'$Sampling \,\, points$', fontsize=20)
    ax.set_ylabel(r'$Ionosphere \,\, correction \,\, residual [m]$',
                  fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=20, loc='best', frameon=False)
    ax.tick_params(labelsize=25, width=2.9)
    # ax.set_title(f"Fit result for cutoff frequenct {fcut}: mu={mu}, std={std}")
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)

    freq_stoc, psd_stoc = welch(iono_corr_hf,
                                10., ('kaiser', 30.),
                                iono_corr.__len__(),
                                scaling='density')
    freq_1b, psd_1b = welch(iono_kbr1b - kaiser(iono_kbr1b, 0.1, 0.02),
                            0.2, ('kaiser', 30.),
                            iono_kbr1b.__len__(),
                            scaling='density')
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.loglog(freq_stoc, np.sqrt(psd_stoc), linewidth=2, label='Ka band')
    ax.loglog(freq_1b, np.sqrt(psd_1b), linewidth=2, label='1b')
    ax.semilogx(np.linspace(0.0001, 5, 10000), 2.62 * np.sqrt(1 + (0.003 / np.linspace(0.0001, 5, 10000) ** 2)) * 1e-6, label='stochastic error requirement')
    ax.set_xlabel(r'$Frequency [Hz]$', fontsize=20)
    ax.set_ylabel(r'$ASD [m/\sqrt{Hz}]$', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=20, loc='best', frameon=False)
    ax.tick_params(labelsize=25, width=2.9)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.show()


def sep_key(df, pos):
    qualflg = df['qualflg']
    return int(qualflg[pos])


def system_noise_iono_residual():
    c = 299792458.
    file_flag = ['KBR1A*', 'KBR1B*', 'USO1B*']
    dtype_kbr1a = {
        'time_intg': np.longdouble,
        'time_frac': np.longdouble,
        'gracefo_id': 'str',
        'prn_id': np.int,
        'ant_id': np.int,
        'prod_flag': np.int,
        'qualflg': 'str',
        'k_phase': np.longdouble,
        'ka_phase': np.longdouble,
        'k_snr': np.longdouble,
        'ka_snr': np.longdouble
    }
    qualflg_list = [
        'k_phase_break', 'ka_phase_break', 'k_cycle_slip', 'ka_cycle_slip',
        'insane', 'missing', 'k_low_snr', 'ka_low_snr'
    ]
    fcut = 0.004 # Hz
    fs_1b = 0.2 # Hz
    fs_1a = 10 # Hz

    kbr1a_c_filename = extract_filenames(file_flag[0], 'C*')
    kbr1a_d_filename = extract_filenames(file_flag[0], 'D*')
    kbr1b_y_filename = extract_filenames(file_flag[1], 'Y*')
    uso1b_c_filename = extract_filenames(file_flag[2], 'C*')
    uso1b_d_filename = extract_filenames(file_flag[2], 'D*')

    kbr1a_c_k_snr = np.zeros(kbr1a_c_filename.__len__())
    kbr1a_d_k_snr = np.zeros(kbr1a_d_filename.__len__())
    kbr1a_c_ka_snr = np.zeros(kbr1a_c_filename.__len__())
    kbr1a_d_ka_snr = np.zeros(kbr1a_d_filename.__len__())
    kbr1a_sn = np.zeros(kbr1a_c_filename.__len__())
    iono_corr = np.zeros(kbr1a_c_filename.__len__())

    for index, (c_1a, d_1a, b, uso_c, uso_d) in enumerate(zip(kbr1a_c_filename, kbr1a_d_filename, kbr1b_y_filename, uso1b_c_filename, uso1b_d_filename)):

        dd_kbr1a_c = dd.read_csv(urlpath=c_1a,
                                 engine='c',
                                 header=None,
                                 names=[
                                     'time_intg', 'time_frac', 'gracefo_id',
                                     'prn_id', 'ant_id', 'prod_flag', 'qualflg',
                                     'k_phase', 'ka_phase', 'k_snr', 'ka_snr'
                                 ],
                                 skiprows=250,
                                 sep='\s+',
                                 dtype=dtype_kbr1a)
        dd_kbr1a_d = dd.read_csv(urlpath=d_1a,
                                 engine='c',
                                 header=None,
                                 names=[
                                     'time_intg', 'time_frac', 'gracefo_id',
                                     'prn_id', 'ant_id', 'prod_flag', 'qualflg',
                                     'k_phase', 'ka_phase', 'k_snr', 'ka_snr'
                                 ],
                                 skiprows=250,
                                 sep='\s+',
                                 dtype=dtype_kbr1a)
        dd_kbr1b = dd.read_csv(urlpath=b,
                               engine='c',
                               header=None,
                               sep='\s+',
                               skiprows=162,
                               dtype=np.longdouble,
                               names=[
                                   'gps_time', 'biased_range', 'range_rate', 'range_accl',
                                   'iono_corr', 'lighttime_err', 'lighttime_rate',
                                   'lighttime_accl', 'ant_centr_corr', 'ant_centr_rate',
                                   'ant_centr_accl', 'k_a_snr', 'ka_a_snr', 'k_b_snr', 'ka_b_snr', 'qualflg'
                               ])
        dd_uso1b_c = dd.read_csv(urlpath=uso_c,
                                 engine='c',
                                 header=None,
                                 sep='\s+',
                                 skiprows=90,
                                 names=[
                                     'gps_time', 'id', 'uso_freq', 'k_freq', 'ka_freq', 'qualflg'
                                 ],
                                 dtype={
                                     'k_freq': np.longdouble,
                                     'ka_freq': np.longdouble},)
        dd_uso1b_d = dd.read_csv(urlpath=uso_d,
                                 engine='c',
                                 header=None,
                                 sep='\s+',
                                 skiprows=90,
                                 names=[
                                     'gps_time', 'id', 'uso_freq', 'k_freq',
                                     'ka_freq', 'qualflg'
                                 ],
                                 dtype={
                                     'k_freq': np.longdouble,
                                     'ka_freq': np.longdouble
                                 },)

        # necessary wrap-up
        dd_kbr1a_c['rcv_time'] = dd_kbr1a_c.time_intg + 1e-6 * dd_kbr1a_c.time_frac
        dd_kbr1a_d['rcv_time'] = dd_kbr1a_d.time_intg + 1e-6 * dd_kbr1a_d.time_frac
        dd_kbr1a_c = dd_kbr1a_c.drop_duplicates(subset=['rcv_time'])
        dd_kbr1a_d = dd_kbr1a_d.drop_duplicates(subset=['rcv_time'])
        dd_kbr1a_d = dd_kbr1a_d[dd_kbr1a_d.rcv_time.isin(dd_kbr1a_c.rcv_time.compute().to_numpy())]

        # extract snr
        kbr1a_c_k_snr[index] = np.nanmean(dd_kbr1b.k_a_snr.compute().to_numpy())
        kbr1a_c_ka_snr[index] = np.nanmean(dd_kbr1b.ka_a_snr.compute().to_numpy())
        kbr1a_d_k_snr[index] = np.nanmean(dd_kbr1b.k_b_snr.compute().to_numpy())
        kbr1a_d_ka_snr[index] = np.nanmean(dd_kbr1b.ka_b_snr.compute().to_numpy())

        # ionosphere-free snr // system noise
        freq_k_c = dd_uso1b_c.k_freq.compute().to_numpy()[0]
        freq_k_d = dd_uso1b_d.k_freq.compute().to_numpy()[0]
        freq_ka_c = dd_uso1b_c.ka_freq.compute().to_numpy()[0]
        freq_ka_d = dd_uso1b_d.ka_freq.compute().to_numpy()[0]
        # effictive frequency
        freq_k_e = np.sqrt(freq_k_c * freq_k_d)
        freq_ka_e = np.sqrt(freq_ka_c * freq_ka_d)

        phase_k = c / (freq_k_c + freq_k_d) * np.sqrt(
            (1. / (2. * np.pi * kbr1a_c_k_snr[index]))**2 +
            (1. / (2. * np.pi * kbr1a_d_k_snr[index]))**2)
        phase_ka = c / (freq_ka_c + freq_ka_d) * np.sqrt(
            (1. / (2. * np.pi * kbr1a_c_ka_snr[index]))**2 +
            (1. / (2. * np.pi * kbr1a_d_ka_snr[index]))**2)

        kbr1a_sn[index] = np.sqrt(((freq_k_e**2 * phase_k)**2 + (freq_ka_e**2 * phase_ka)**2) /
                                  (freq_k_e**2 - freq_ka_e**2)**2) * np.sqrt(1. / 5.)

        # low-pass filter ionosphere correction
        iono_corr[index] = np.mean(dd_kbr1b.iono_corr.compute().to_numpy() - kaiser(dd_kbr1b.iono_corr.compute().to_numpy(), 0.1, 0.02))


    plt.style.use(['science', 'no-latex', 'high-vis'])
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.scatter(np.arange(kbr1a_c_filename.__len__()), kbr1a_c_k_snr, linewidth=2, label='k_c')
    # ax.scatter(np.arange(kbr1a_c_filename.__len__()), kbr1a_c_ka_snr, linewidth=2, label='ka_c')
    # ax.scatter(np.arange(kbr1a_c_filename.__len__()), kbr1a_d_k_snr, linewidth=2, label='k_d')
    # ax.scatter(np.arange(kbr1a_c_filename.__len__()), kbr1a_d_ka_snr, linewidth=2, label='ka_d')
    # ax.set_xlabel(r'$Day$', fontsize=20)
    # ax.set_ylabel(r'$KBR1A SNR$', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.legend(fontsize=20, loc='best', frameon=False)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    #
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.scatter(np.arange(kbr1a_c_filename.__len__()),
    #            kbr1a_sn,
    #            linewidth=2,
    #            label='kbr system noise')
    # ax.set_xlabel(r'$Day$', fontsize=20)
    # ax.set_ylabel(r'$KBR1A SNR$', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.legend(fontsize=20, loc='best', frameon=False)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    #
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.scatter(np.arange(kbr1a_c_filename.__len__()),
    #            iono_corr,
    #            linewidth=2,
    #            label='ionosphere free correction residual')
    # ax.set_xlabel(r'$Day$', fontsize=20)
    # ax.set_ylabel(r'$KBR1A SNR$', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.legend(fontsize=20, loc='best', frameon=False)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(kbr1a_sn,
               iono_corr,
               linewidth=2,
               label='ionosphere free correction residual')
    ax.set_xlabel(r'$KBR1A \,\, system \,\, noise$', fontsize=20)
    ax.set_ylabel(r'$Ionosphere \,\, correction \,\, residual$', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=20, loc='best', frameon=False)
    ax.tick_params(labelsize=25, width=2.9)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.show()
    print(iono_corr, kbr1a_sn)
    print(pearsonr(iono_corr, kbr1a_sn))


def kaiser(x, fq, cutoff_hz, ripple_db=300.):
    # the desired width of the transition from pass to stop
    width = 0.12 / fq

    # the desired attenuation in the stop band in db: ripple_db

    # compute the kaiser parameter for the fir filter
    n, beta = kaiserord(ripple_db, width)
    print('The length of the lowpass filter is ', n, '.')

    # use firwin with a kaiser window
    taps = firwin(n,
                  cutoff_hz,
                  window=('kaiser', beta),
                  pass_zero='lowpass',
                  nyq=fq)

    # use filtfilt to filter x with the fir filter
    filtered_x = filtfilt(taps, 1.0, x)

    return filtered_x


def outlier():
    file_flag = ['KBR1A*', 'KBR1B*']
    dtype_kbr1a = {
        'time_intg': np.longdouble,
        'time_frac': np.longdouble,
        'gracefo_id': 'str',
        'prn_id': np.int,
        'ant_id': np.int,
        'prod_flag': np.int,
        'qualflg': 'str',
        'k_phase': np.longdouble,
        'ka_phase': np.longdouble,
        'k_snr': np.longdouble,
        'ka_snr': np.longdouble
    }
    qualflg_list = [
        'k_phase_break', 'ka_phase_break', 'k_cycle_slip', 'ka_cycle_slip',
        'insane', 'missing', 'k_low_snr', 'ka_low_snr'
    ]

    kbr1a_c_filename = extract_filenames(file_flag[0], 'C*')
    kbr1a_d_filename = extract_filenames(file_flag[0], 'D*')
    kbr1b_y_filename = extract_filenames(file_flag[1], 'Y*')

    dd_kbr1a_c = dd.read_csv(urlpath=kbr1a_c_filename,
                             engine='c',
                             header=None,
                             names=[
                                 'time_intg', 'time_frac', 'gracefo_id',
                                 'prn_id', 'ant_id', 'prod_flag', 'qualflg',
                                 'k_phase', 'ka_phase', 'k_snr', 'ka_snr'
                             ],
                             skiprows=235,
                             sep='\s+',
                             dtype=dtype_kbr1a)
    dd_kbr1a_d = dd.read_csv(urlpath=kbr1a_d_filename,
                             engine='c',
                             header=None,
                             names=[
                                 'time_intg', 'time_frac', 'gracefo_id',
                                 'prn_id', 'ant_id', 'prod_flag', 'qualflg',
                                 'k_phase', 'ka_phase', 'k_snr', 'ka_snr'
                             ],
                             skiprows=235,
                             sep='\s+',
                             dtype=dtype_kbr1a)
    dd_kbr1b = dd.read_csv(urlpath=kbr1b_y_filename,
                           engine='c',
                           header=None,
                           sep='\s+',
                           skiprows=162,
                           dtype=np.longdouble,
                           names=[
                               'gps_time', 'biased_range', 'range_rate', 'range_accl',
                               'iono_corr', 'lighttime_err', 'lighttime_rate',
                               'lighttime_accl', 'ant_centr_corr', 'ant_centr_rate',
                               'ant_centr_accl', 'k_a_snr', 'ka_a_snr', 'k_b_snr', 'ka_b_snr', 'qualflg'
                           ])

    # necessary wrap-up
    dd_kbr1a_c['rcv_time'] = dd_kbr1a_c.time_intg + 1e-6 * dd_kbr1a_c.time_frac
    dd_kbr1a_d['rcv_time'] = dd_kbr1a_d.time_intg + 1e-6 * dd_kbr1a_d.time_frac
    dd_kbr1a_c = dd_kbr1a_c.drop_duplicates(subset=['rcv_time'])
    dd_kbr1a_d = dd_kbr1a_d.drop_duplicates(subset=['rcv_time'])
    dd_kbr1a_d = dd_kbr1a_d[dd_kbr1a_d.rcv_time.isin(dd_kbr1a_c.rcv_time.compute().to_numpy())]
    plt.scatter(dd_kbr1a_c['rcv_time'].compute().to_numpy(), dd_kbr1a_c['k_snr'].compute().to_numpy())
    plt.show()

    # for index, _ in enumerate(np.arange(8)):
    #     dd_kbr1a_c[qualflg_list[index]] = dd_kbr1a_c.apply(
    #         sep_key, meta=dd_kbr1a_c['qualflg'], pos=-index - 1, axis=1)
    #     dd_kbr1a_d[qualflg_list[index]] = dd_kbr1a_d.apply(
    #         sep_key, meta=dd_kbr1a_d['qualflg'], pos=-index - 1, axis=1)

    # # output a file
    # dd_kbr1a_c[qualflg_list].to_csv('..//output//kbr1a_flags_c.txt',
    #                                 single_file=True,
    #                                 mode='wt',
    #                                 index=False)
    # dd_kbr1a_d[qualflg_list].to_csv('..//output//kbr1a_flags_d.txt',
    #                                 single_file=True,
    #                                 mode='wt',
    #                                 index=False)


    # outlier_dict_c = create_dict(qualflg_list)
    # outlier_dict_d = create_dict(qualflg_list)
    # for i, qualflg in enumerate(qualflg_list):
    #     outlier_dict_c[qualflg] = dd_kbr1a_c[dd_kbr1a_c[qualflg] == 1]['rcv_time'].compute().to_numpy()
    #     outlier_dict_d[qualflg] = dd_kbr1a_d[dd_kbr1a_d[qualflg] == 1]['rcv_time'].compute().to_numpy()
    # print(outlier_dict_c)


def extract_filenames(file_flag, id):
    gracefo_dataset_dir = ''
    if platform.system() == 'Linux':
        gracefo_dataset_dir = '/home/reuben/windows_disk/e/lhsProgrammes/gracefo_dataset'
    elif platform.system() == 'Windows':
        gracefo_dataset_dir = 'E:/lhsProgrammes/gracefo_dataset'
    suffix = '.txt'
    temp = glob.glob(gracefo_dataset_dir + '/**' + '/' + file_flag + id + suffix)
    temp.sort()
    return temp


def create_dict(arg_in):
    dict = {}
    for i, flag in enumerate(arg_in):
        dict[flag] = []
    return dict


if __name__ == '__main__':
    iono_corr()
    # outlier()
    # system_noise_iono_residual()
