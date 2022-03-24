# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import dask.dataframe as dd
from matplotlib import ticker


def main():
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei"] 
    plt.rcParams["axes.unicode_minus"]=False
    crn_range = np.loadtxt('..//output//crn_range_residual_2019-01-01.txt', dtype=np.longdouble, delimiter=',')
    crn_rate = np.loadtxt('..//output//crn_rate_residual_2019-01-01.txt', dtype=np.longdouble, delimiter=',')
    crn_accl = np.loadtxt('..//output//crn_accl_residual_2019-01-01.txt', dtype=np.longdouble, delimiter=',')
    kaiser_range = np.loadtxt('..//output//kaiser_range_residual_2019-01-01.txt', dtype=np.longdouble, delimiter=',')
    kaiser_rate = np.loadtxt('..//output//kaiser_rate_residual_2019-01-01.txt', dtype=np.longdouble, delimiter=',')
    kaiser_accl = np.loadtxt('..//output//kaiser_accl_residual_2019-01-01.txt', dtype=np.longdouble, delimiter=',')

    fs = 0.2

    freq_range_jpl, psd_range_jpl = welch(crn_range[:, 0], fs, ('kaiser', 30.), crn_range[:, 0].__len__(), scaling='density')
    freq_rate_jpl, psd_rate_jpl = welch(crn_rate[:, 0], fs, ('kaiser', 30.), crn_rate[:, 0].__len__(), scaling='density')
    freq_accl_jpl, psd_accl_jpl = welch(crn_accl[:, 0], fs, ('kaiser', 30.), crn_accl[:, 0].__len__(), scaling='density')

    freq_range_cau, psd_range_cau = welch(crn_range[:, 1], fs, ('kaiser', 30.), crn_range[:, 0].__len__(), scaling='density')
    freq_rate_cau, psd_rate_cau = welch(crn_rate[:, 1], fs, ('kaiser', 30.), crn_rate[:, 0].__len__(), scaling='density')
    freq_accl_cau, psd_accl_cau = welch(crn_accl[:, 1], fs, ('kaiser', 30.), crn_accl[:, 0].__len__(), scaling='density')

    freq_range_residual, psd_range_residual = welch(crn_range[:, 2], fs, ('kaiser', 30.), crn_range[:, 0].__len__(), scaling='density')
    freq_rate_residual, psd_rate_residual = welch(crn_rate[:, 2], fs, ('kaiser', 30.), crn_rate[:, 0].__len__(), scaling='density')
    freq_accl_residual, psd_accl_residual = welch(crn_accl[:, 2], fs, ('kaiser', 30.), crn_accl[:, 0].__len__(), scaling='density')

    freq_range_kaiser, psd_range_kaiser = welch(kaiser_range[:, 1], fs, ('kaiser', 30.), kaiser_range[:, 0].__len__(), scaling='density')
    freq_rate_kaiser, psd_rate_kaiser = welch(kaiser_rate[:, 1], fs, ('kaiser', 30.), kaiser_rate[:, 0].__len__(), scaling='density')
    freq_accl_kaiser, psd_accl_kaiser = welch(kaiser_accl[:, 1], fs, ('kaiser', 30.), kaiser_accl[:, 0].__len__(), scaling='density')

    freq_range_kresidual, psd_range_kresidual = welch(kaiser_range[:, 2], fs, ('kaiser', 30.), kaiser_range[:, 0].__len__(), scaling='density')
    freq_rate_kresidual, psd_rate_kresidual = welch(kaiser_rate[:, 2], fs, ('kaiser', 30.), kaiser_rate[:, 0].__len__(), scaling='density')
    freq_accl_kresidual, psd_accl_kresidual = welch(kaiser_accl[:, 2], fs, ('kaiser', 30.), kaiser_accl[:, 0].__len__(), scaling='density')

    # plt.style.use(['science', 'no-latex', "cjk-sc-font"])
    # crn range time series
    fig = plt.figure(figsize=(20, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(1, 2, 2)
    ax1.plot(np.linspace(0, 86400, crn_range.__len__()), crn_range[:, 1], linewidth=3, label='UBN_KBR_RL01', marker='o', color='xkcd:aqua', alpha=0.5)
    ax1.plot(np.linspace(0, 86400, crn_range.__len__()), crn_range[:, 0], linewidth=2, label='JPL_KBR_RL04', color='xkcd:coral', linestyle='dashed')
    ax1.yaxis.get_offset_text().set_fontsize(24)
    ax1.tick_params(labelsize=20, width=2.9)
    ax1.set_ylim(crn_range[:, 1].min() - 100, crn_range[:, 1].max() + 200)
    ax1.set_ylabel('有偏星间距 [m]', fontsize=20)
    ax1.legend(fontsize=15, loc='upper right', frameon=False)
    ax1.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax1.text(0.01, 78450, "(a)", fontsize=15)
    ax2.plot(np.linspace(0, 86400, crn_range.__len__()),
               crn_range[:, 2],
               linewidth=1,
               label='residual')
    ax2.tick_params(labelsize=20, width=2.9)
    ax2.yaxis.get_offset_text().set_fontsize(20)
    ax2.set_xlabel('自2019-01-01T00:00:00开始GPS时', fontsize=20)
    ax2.set_ylabel('星间距残差 [m]', fontsize=20)
    ax2.text(0, -2e-8, "(b)", fontsize=15)
    ax2.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax3.loglog(freq_range_cau,
                 np.sqrt(psd_range_cau),
                 linewidth=4,
                 label='UBN_KBR_RL01',
                 color='xkcd:aqua',)
    ax3.loglog(freq_range_jpl,
                 np.sqrt(psd_range_jpl),
                 linewidth=1.5,
                 label='JPL_KBR_RL04',
                 color='xkcd:coral',
                 linestyle='dashed')
    ax3.loglog(freq_range_residual,
                 np.sqrt(psd_range_residual),
                 linewidth=2,
                 label='残差')
    ax3.text(1.5e-5, 1e-10, "(c)", fontsize=15)
    ax3.tick_params(labelsize=25, width=2.9)
    ax3.set_xlabel('频率 [Hz]', fontsize=20)
    ax3.yaxis.get_offset_text().set_fontsize(24)
    ax3.set_ylabel(r'ASD [m/$\sqrt{Hz}$]', fontsize=20)
    ax3.legend(fontsize=15, loc='best', frameon=False)
    ax3.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    # ax[2].gca().spines['left'].set_linewidth(2)
    # ax[2].gca().spines['top'].set_linewidth(2)
    # ax[2].gca().spines['right'].set_linewidth(2)
    # ax[2].gca().spines['bottom'].set_linewidth(2)
    plt.tight_layout()
    plt.show()
    fig.savefig('..//images//crn_range_time_series.png', dpi=500)

    fig = plt.figure(figsize=(20, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(1, 2, 2)
    ax1.plot(np.linspace(0, 86400, crn_rate.__len__()), crn_rate[:, 1], linewidth=3, label='UBN_KBR_RL01', marker='o', color='xkcd:aqua', alpha=0.5)
    ax1.plot(np.linspace(0, 86400, crn_rate.__len__()), crn_rate[:, 0], linewidth=2, label='JPL_KBR_RL04', color='xkcd:coral', linestyle='dashed')
    ax1.yaxis.get_offset_text().set_fontsize(24)
    ax1.tick_params(labelsize=20, width=2.9)
    ax1.set_ylim(crn_rate[:, 1].min() - 0.3, crn_rate[:, 1].max() + 0.45)
    ax1.set_ylabel('星间变率 [m/s]', fontsize=20)
    ax1.legend(fontsize=15, loc='upper right', frameon=False)
    ax1.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax1.text(0.01, -0.7, "(a)", fontsize=15)
    ax2.plot(np.linspace(0, 86400, crn_rate.__len__()),
               crn_rate[:, 2],
               linewidth=1,
               label='residual')
    ax2.tick_params(labelsize=20, width=2.9)
    ax2.yaxis.get_offset_text().set_fontsize(20)
    ax2.set_xlabel('自2019-01-01T00:00:00开始GPS时', fontsize=20)
    ax2.set_ylabel('星间变量残差 [m/s]', fontsize=20)
    ax2.text(0, -8e-8, "(b)", fontsize=15)
    ax2.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax3.loglog(freq_rate_cau,
                 np.sqrt(psd_rate_cau),
                 linewidth=4,
                 label='UBN_KBR_RL01',
                 color='xkcd:aqua',)
    ax3.loglog(freq_rate_jpl,
                 np.sqrt(psd_rate_jpl),
                 linewidth=1.5,
                 label='JPL_KBR_RL04',
                 color='xkcd:coral',
                 linestyle='dashed')
    ax3.loglog(freq_rate_residual,
                 np.sqrt(psd_rate_residual),
                 linewidth=2,
                 label='残差')
    ax3.text(1.5e-5, 1e-10, "(c)", fontsize=15)
    ax3.tick_params(labelsize=25, width=2.9)
    ax3.set_xlabel('频率 [Hz]', fontsize=20)
    ax3.yaxis.get_offset_text().set_fontsize(24)
    ax3.set_ylabel(r'ASD [m/s/$\sqrt{Hz}$]', fontsize=20)
    ax3.legend(fontsize=15, loc='best', frameon=False)
    ax3.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.tight_layout()
    # ax[2].gca().spines['left'].set_linewidth(2)
    # ax[2].gca().spines['top'].set_linewidth(2)
    # ax[2].gca().spines['right'].set_linewidth(2)
    # ax[2].gca().spines['bottom'].set_linewidth(2)
    plt.show()
    fig.savefig('..//images//crn_rate_time_series.png', dpi=500)

    fig = plt.figure(figsize=(20, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(1, 2, 2)
    ax1.plot(np.linspace(0, 86400, crn_accl.__len__()), crn_accl[:, 1], linewidth=3, label='UBN_KBR_RL01', marker='o', color='xkcd:aqua', alpha=0.5)
    ax1.plot(np.linspace(0, 86400, crn_accl.__len__()), crn_accl[:, 0], linewidth=2, label='JPL_KBR_RL04', color='xkcd:coral', linestyle='dashed')
    ax1.yaxis.get_offset_text().set_fontsize(24)
    ax1.tick_params(labelsize=20, width=2.9)
    ax1.set_ylim(crn_accl[:, 1].min() - 2e-4, crn_accl[:, 1].max() + 6e-4)
    ax1.set_ylabel('星间加速度 [m/s]', fontsize=20)
    ax1.legend(fontsize=15, loc='upper right', frameon=False)
    ax1.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax1.text(0.01, -6e-4, "(a)", fontsize=15)
    ax2.plot(np.linspace(0, 86400, crn_accl.__len__()),
             crn_accl[:, 2],
             linewidth=1,
             label='residual')
    ax2.tick_params(labelsize=20, width=2.9)
    ax2.yaxis.get_offset_text().set_fontsize(20)
    ax2.set_xlabel('自2019-01-01T00:00:00开始GPS时', fontsize=20)
    ax2.set_ylabel('星间加速度残差 [m/s]', fontsize=20)
    ax2.text(0, -4e-8, "(b)", fontsize=15)
    ax2.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax3.loglog(freq_accl_cau,
                 np.sqrt(psd_accl_cau),
                 linewidth=4,
                 label='UBN_KBR_RL01',
                 color='xkcd:aqua',)
    ax3.loglog(freq_accl_jpl,
                 np.sqrt(psd_accl_jpl),
                 linewidth=1.5,
                 label='JPL_KBR_RL04',
                 color='xkcd:coral',
                 linestyle='dashed')
    ax3.loglog(freq_accl_residual,
                 np.sqrt(psd_accl_residual),
                 linewidth=2,
                 label='残差')
    ax3.text(1.5e-5, 1e-12, "(c)", fontsize=15)
    ax3.tick_params(labelsize=25, width=2.9)
    ax3.set_xlabel('频率 [Hz]', fontsize=20)
    ax3.yaxis.get_offset_text().set_fontsize(24)
    ax3.set_ylabel(r'ASD [m/$s^2/\sqrt{Hz}$]', fontsize=20)
    ax3.legend(fontsize=15, loc='best', frameon=False)
    ax3.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.tight_layout()
    # ax[2].gca().spines['left'].set_linewidth(2)
    # ax[2].gca().spines['top'].set_linewidth(2)
    # ax[2].gca().spines['right'].set_linewidth(2)
    # ax[2].gca().spines['bottom'].set_linewidth(2)
    plt.show()
    fig.savefig('..//images//crn_accl_time_series.png', dpi=500)
    exit()

    # crn rate time series
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].plot(np.linspace(0, 86400, crn_rate.__len__()), crn_rate[:, 1], linewidth=2, label='UBN', marker='o', color='xkcd:aqua', alpha=0.5)
    ax[0].plot(np.linspace(0, 86400, crn_rate.__len__()), crn_rate[:, 0], linewidth=2, label='JPL')
    ax[0].tick_params(labelsize=20, width=2.9)
    ax[0].set_xlabel('Sampling Points [5Hz]', fontsize=20)
    ax[0].set_ylabel(r'Range Rate [m/s]', fontsize=20)
    ax[0].legend(fontsize=15, loc='best')
    ax[0].yaxis.get_offset_text().set_fontsize(24)
    ax[0].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax[1].plot(np.linspace(0, 86400, crn_rate.__len__()),
               crn_rate[:, 2],
               linewidth=1,
               label='residual')
    ax[1].tick_params(labelsize=20, width=2.9)
    ax[1].set_xlabel('Sampling Points [5Hz]', fontsize=20)
    ax[1].set_ylabel(r'Range rate residual [m/s]', fontsize=20)
    ax[1].yaxis.get_offset_text().set_fontsize(24)
    ax[1].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    fig.savefig('..//images//crn_rate_time_series.png', dpi=500)

    exit()

    # crn accl time series
    fig, ax = plt.subplots(2, 1, figsize=(50, 25))
    ax[0].plot(np.linspace(0, 86400, crn_accl.__len__()), crn_accl[:, 1], linewidth=2, label='UBN', marker='o', color='xkcd:aqua', alpha=0.5)
    ax[0].plot(np.linspace(0, 86400, crn_accl.__len__()), crn_accl[:, 0], linewidth=2, label='JPL')
    ax[0].tick_params(labelsize=20, width=2.9)
    ax[0].set_xlabel('Sampling Points [5Hz]', fontsize=20)
    ax[0].set_ylabel(r'Range accelaration [m/s]', fontsize=20)
    ax[0].legend(fontsize=15, loc='best')
    ax[0].yaxis.get_offset_text().set_fontsize(24)
    ax[0].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax[1].plot(np.linspace(0, 86400, crn_accl.__len__()),
               crn_accl[:, 2],
               linewidth=1,
               label='residual')
    ax[1].tick_params(labelsize=20, width=2.9)
    ax[1].set_xlabel('Sampling Points [5Hz]', fontsize=20)
    ax[1].set_ylabel(r'Range accelaration residual [m/s]', fontsize=20)
    ax[1].yaxis.get_offset_text().set_fontsize(24)
    ax[1].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    fig.savefig('..//images//crn_accl_time_series.png', dpi=500)

    # kaiser range time series
    fig, ax = plt.subplots(2, 1, figsize=(50, 25))
    ax[0].plot(np.linspace(0, 86400, kaiser_range.__len__()), kaiser_range[:, 1], linewidth=2, label='UBN', marker='o', color='xkcd:lavender', alpha=0.5)
    ax[0].plot(np.linspace(0, 86400, kaiser_range.__len__()), kaiser_range[:, 0], linewidth=2, label='JPL')
    ax[0].tick_params(labelsize=20, width=2.9)
    ax[0].set_xlabel('Sampling Points [5Hz]', fontsize=20)
    ax[0].set_ylabel(r'Intersatelite range [m]', fontsize=20)
    ax[0].legend(fontsize=15, loc='best')
    ax[0].yaxis.get_offset_text().set_fontsize(24)
    ax[0].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax[1].plot(np.linspace(0, 86400, kaiser_range.__len__()),
               kaiser_range[:, 2],
               linewidth=1,
               label='residual')
    ax[1].tick_params(labelsize=20, width=2.9)
    ax[1].set_xlabel('Sampling Points [5Hz]', fontsize=20)
    ax[1].set_ylabel(r'Intersatelite range residual [m]', fontsize=20)
    ax[1].yaxis.get_offset_text().set_fontsize(24)
    ax[1].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    fig.savefig('..//images//kaiser_range_time_series.png', dpi=500)

    # kaiser rate time series
    fig, ax = plt.subplots(2, 1, figsize=(50, 25))
    ax[0].plot(np.linspace(0, 86400, kaiser_rate.__len__()), kaiser_rate[:, 1], linewidth=2, label='UBN', marker='o', color='xkcd:lavender', alpha=0.5)
    ax[0].plot(np.linspace(0, 86400, kaiser_rate.__len__()), kaiser_rate[:, 0], linewidth=2, label='JPL')
    ax[0].tick_params(labelsize=20, width=2.9)
    ax[0].set_xlabel('Sampling Points [5Hz]', fontsize=20)
    ax[0].set_ylabel(r'Range Rate [m/s]', fontsize=20)
    ax[0].legend(fontsize=15, loc='best')
    ax[0].yaxis.get_offset_text().set_fontsize(24)
    ax[0].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax[1].plot(np.linspace(0, 86400, kaiser_rate.__len__()),
               kaiser_rate[:, 2],
               linewidth=1,
               label='residual')
    ax[1].tick_params(labelsize=20, width=2.9)
    ax[1].set_xlabel('Sampling Points [5Hz]', fontsize=20)
    ax[1].set_ylabel(r'Range rate residual [m/s]', fontsize=20)
    ax[1].yaxis.get_offset_text().set_fontsize(24)
    ax[1].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    fig.savefig('..//images//kaiser_rate_time_series.png', dpi=500)

    # kaiser accl time series
    fig, ax = plt.subplots(2, 1, figsize=(50, 25))
    ax[0].plot(np.linspace(0, 86400, kaiser_accl.__len__()), kaiser_accl[:, 1], linewidth=2, label='UBN', marker='o', color='xkcd:lavender', alpha=0.5)
    ax[0].plot(np.linspace(0, 86400, kaiser_accl.__len__()), kaiser_accl[:, 0], linewidth=2, label='JPL')
    ax[0].tick_params(labelsize=20, width=2.9)
    ax[0].set_xlabel('Sampling Points [5Hz]', fontsize=20)
    ax[0].set_ylabel(r'Range accelaration [m/s]', fontsize=20)
    ax[0].legend(fontsize=15, loc='best')
    ax[0].yaxis.get_offset_text().set_fontsize(24)
    ax[0].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax[1].plot(np.linspace(0, 86400, kaiser_accl.__len__()),
               kaiser_accl[:, 2],
               linewidth=1,)
    ax[1].tick_params(labelsize=20, width=2.9)
    ax[1].set_xlabel('Sampling Points [5Hz]', fontsize=20)
    ax[1].set_ylabel(r'Range accelaration residual [m/s]', fontsize=20)
    ax[1].yaxis.get_offset_text().set_fontsize(24)
    ax[1].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    fig.savefig('..//images//kaiser_accl_time_series.png', dpi=500)
    # plt.gca().spines['left'].set_linewidth(2)
    # plt.gca().spines['top'].set_linewidth(2)
    # plt.gca().spines['right'].set_linewidth(2)
    # plt.gca().spines['bottom'].set_linewidth(2)
    # plt.show()

    plt.style.use(['science', 'no-latex', 'high-vis'])
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.loglog(freq_range_jpl, np.sqrt(psd_range_jpl), linewidth=2, label=r'$KBR1B-crn-JPL$')
    plt.loglog(freq_range_cau, np.sqrt(psd_range_cau), linewidth=2, label=r'$KBR1B-crn-UBN$')
    plt.loglog(freq_range_residual, np.sqrt(psd_range_residual), linewidth=2, label=r'$residual$')
    plt.tick_params(labelsize=25, width=2.9)
    plt.xlabel('Frequency [Hz]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    plt.ylabel(r'$Range \, \, ASD [m/\sqrt{Hz}]$', fontsize=20)
    plt.legend(fontsize=15, loc='best')
    plt.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    fig.savefig('..//images//crn_range_psd.png', dpi=500)

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.loglog(freq_rate_jpl, np.sqrt(psd_rate_jpl), linewidth=2, label=r'$KBR1B-crn-JPL$')
    plt.loglog(freq_rate_cau, np.sqrt(psd_rate_cau), linewidth=2, label=r'$KBR1B-crn-UBN$')
    plt.loglog(freq_rate_residual, np.sqrt(psd_rate_residual), linewidth=2, label=r'$residual$')
    plt.tick_params(labelsize=25, width=2.9)
    plt.xlabel('Frequency [Hz]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    plt.ylabel(r'$Range Rate \, \, ASD [m/s/\sqrt{Hz}]$', fontsize=20)
    plt.legend(fontsize=15, loc='best')
    plt.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    fig.savefig('..//images//crn_rate_psd.png', dpi=500)

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.loglog(freq_accl_jpl, np.sqrt(psd_accl_jpl), linewidth=2, label=r'$KBR1B-crn-JPL$')
    plt.loglog(freq_accl_cau, np.sqrt(psd_accl_cau), linewidth=2, label=r'$KBR1B-crn-UBN$')
    plt.loglog(freq_accl_residual, np.sqrt(psd_accl_residual), linewidth=2, label=r'$residual$')
    plt.tick_params(labelsize=25, width=2.9)
    plt.xlabel('Frequency [Hz]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    plt.ylabel(r'$Range Accelaration \, \, ASD [m/s^2/\sqrt{Hz}]$', fontsize=20)
    plt.legend(fontsize=15, loc='best')
    plt.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    fig.savefig('..//images//crn_accl_psd.png', dpi=500)

    plt.style.use(['science', 'no-latex', 'high-vis'])
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.loglog(freq_range_jpl,
               np.sqrt(psd_range_jpl),
               linewidth=2,
               label=r'$KBR1B-crn-JPL$')
    plt.loglog(freq_range_kaiser,
               np.sqrt(psd_range_kaiser),
               linewidth=2,
               label=r'$KBR1B-Kaiser-UBN$')
    plt.loglog(freq_range_kresidual,
               np.sqrt(psd_range_kresidual),
               linewidth=2,
               label=r'$residual$')
    plt.tick_params(labelsize=25, width=2.9)
    plt.xlabel('Frequency [Hz]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    plt.ylabel(r'$Range \, \, ASD [m/\sqrt{Hz}]$', fontsize=20)
    plt.legend(fontsize=15, loc='best')
    plt.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    fig.savefig('..//images//kaiser_range_psd.png', dpi=500)

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.loglog(freq_rate_jpl,
               np.sqrt(psd_rate_jpl),
               linewidth=2,
               label=r'$KBR1B-crn-JPL$')
    plt.loglog(freq_rate_kaiser,
               np.sqrt(psd_rate_kaiser),
               linewidth=2,
               label=r'$KBR1B-kaiser-UBN$')
    plt.loglog(freq_rate_kresidual,
               np.sqrt(psd_rate_kresidual),
               linewidth=2,
               label=r'$residual$')
    plt.tick_params(labelsize=25, width=2.9)
    plt.xlabel('Frequency [Hz]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    plt.ylabel(r'$Range Rate \, \, ASD [m/s/\sqrt{Hz}]$', fontsize=20)
    plt.legend(fontsize=15, loc='best')
    plt.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    fig.savefig('..//images//kaiser_rate_psd.png', dpi=500)

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.loglog(freq_accl_jpl,
               np.sqrt(psd_accl_jpl),
               linewidth=2,
               label=r'$KBR1B-crn-JPL$')
    plt.loglog(freq_accl_kaiser,
               np.sqrt(psd_accl_kaiser),
               linewidth=2,
               label=r'$KBR1B-kaiser-UBN$')
    plt.loglog(freq_accl_kresidual,
               np.sqrt(psd_accl_kresidual),
               linewidth=2,
               label=r'$residual$')
    plt.tick_params(labelsize=25, width=2.9)
    plt.xlabel('Frequency [Hz]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    plt.ylabel(r'$Range Accelaration \, \, ASD [m/s^2/\sqrt{Hz}]$',
               fontsize=20)
    plt.legend(fontsize=15, loc='best')
    plt.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    fig.savefig('..//images//kaiser_accl_psd.png', dpi=500)
    plt.show()


def tof():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  #设置字体
    plt.rcParams["axes.unicode_minus"] = False
    tof = np.loadtxt('..//output//tof_2019-01-01.txt')
    dd_KBR1B = dd.read_csv(
        urlpath='..//input//KBR1B_2019-01-01_Y_04.txt', engine='c', header=None,
        sep='\s+', skiprows=162, dtype=np.longdouble, names=[
            'gps_time', 'biased_range', 'range_rate', 'range_accl', 'iono_corr', 'lighttime_corr', 'lighttime_rate',
            'lighttime_accl', 'ant_centr_corr', 'ant_centr_rate', 'ant_centr_accl', 'k_a_snr',
            'ka_a_snr', 'k_b_snr', 'ka_b_snr', 'qualflg'
        ]
    )

    tof_range_1b = dd_KBR1B.lighttime_corr.compute().to_numpy()
    fs = 0.2
    freq_tof_jpl, psd_tof_jpl = welch(tof_range_1b[92: 16092], fs, ('kaiser', 30.), tof_range_1b[92: 16092].__len__(), scaling='density')
    freq_tof_chd, psd_tof_chd = welch(tof[460: 80460: 5], fs, ('kaiser', 30.), tof[460: 80460: 5].__len__(), scaling='density')
    freq_tof_res, psd_tof_res = welch(tof[460: 80460: 5] - tof_range_1b[92: 16092], fs, ('kaiser', 30.), tof[460: 80460: 5].__len__(), scaling='density')

    fig = plt.figure(figsize=(20, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    # ax3 = plt.subplot(1, 2, 2)
    ax1.plot(np.linspace(0, 86400, tof[460: 80460: 5].__len__()),
             tof[460: 80460: 5],
             linewidth=3,
             label='UBN_KBR_RL01',
             marker='o',
             color='xkcd:aqua',
             alpha=0.5)
    ax1.plot(np.linspace(0, 86400, tof[460: 80460: 5].__len__()),
             tof_range_1b[92: 16092],
             linewidth=2,
             label='JPL_KBR_RL04',
             color='xkcd:coral',
             linestyle='dashed')
    ax1.yaxis.get_offset_text().set_fontsize(24)
    ax1.tick_params(labelsize=20, width=2.9)
    ax1.set_ylim(tof_range_1b[92:16092].min() - 2e-4,
                 tof_range_1b[92:16092].max() + 2e-4)
    ax1.set_ylabel('飞行时改正 [m]', fontsize=20)
    ax1.legend(fontsize=15, loc='upper right', frameon=False)
    ax1.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax1.text(0.01, -7e-4, "(a)", fontsize=15)
    ax2.plot(np.linspace(0, 86400, tof[460: 80460: 5].__len__()),
             tof[460: 80460: 5] - tof_range_1b[92: 16092],
             linewidth=1,
             label='residual')
    ax2.tick_params(labelsize=20, width=2.9)
    ax2.yaxis.get_offset_text().set_fontsize(20)
    ax2.set_xlabel('自2019-01-01T00:00:00开始GPS时', fontsize=20)
    ax2.set_ylabel('飞行时间改正残差 [m/s]', fontsize=20)
    ax2.text(0, -4e-9, "(b)", fontsize=15)
    ax2.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    # ax3.loglog(
    #     freq_tof_chd,
    #     np.sqrt(psd_tof_chd),
    #     linewidth=4,
    #     label='UBN_KBR_RL01',
    #     color='xkcd:aqua',
    # )
    # ax3.loglog(freq_tof_jpl,
    #            np.sqrt(psd_tof_jpl),
    #            linewidth=1.5,
    #            label='JPL_KBR_RL04',
    #            color='xkcd:coral',
    #            linestyle='dashed')
    # ax3.loglog(freq_tof_res,
    #            np.sqrt(psd_tof_res),
    #            linewidth=2,
    #            label='残差')
    # ax3.text(1.5e-5, 1e-11, "(c)", fontsize=15)
    # ax3.tick_params(labelsize=25, width=2.9)
    # ax3.set_xlabel('频率 [Hz]', fontsize=20)
    # ax3.yaxis.get_offset_text().set_fontsize(24)
    # ax3.set_ylabel(r'ASD [m/$\sqrt{Hz}$]', fontsize=20)
    # ax3.legend(fontsize=15, loc='best', frameon=False)
    # ax3.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.tight_layout()
    # ax[2].gca().spines['left'].set_linewidth(2)
    # ax[2].gca().spines['top'].set_linewidth(2)
    # ax[2].gca().spines['right'].set_linewidth(2)
    # ax[2].gca().spines['bottom'].set_linewidth(2)
    plt.show()
    fig.savefig('..//images//tof_psd_xxx.png', dpi=500)


def kaiser_compare():
    dd_KBR1B_crnfil = dd.read_csv(
        urlpath='..//input//KBR1B_2019-01-01_Y_04.txt',
        sep='\s+',
        header=None,
        skiprows=162,
        dtype=np.longdouble,
        engine='c',
        storage_options=dict(auto_mkdir=False))
    dd_KBR1A = dd.read_csv(
        urlpath='..//output//DOWR1A_2019-01-01_Y_04.txt',
        sep='\s+',
        header=None,
        skiprows=0,
        dtype=np.longdouble,
        engine='c',
        storage_options=dict(auto_mkdir=False))
    ay_KBR1B_kaiser = np.loadtxt(
        '..//output//kaiser_10hz2019-01-01.txt',
        dtype=np.longdouble,
        delimiter=','
    )

    ay_KBR1B_crnfil = dd_KBR1B_crnfil.compute().to_numpy()
    ay_KBR1A = dd_KBR1A.compute().to_numpy()

    freq_crn, psd_crn = welch(ay_KBR1B_crnfil[:, 1], 0.2, ('kaiser', 30.), ay_KBR1B_crnfil[:, 0].__len__(), scaling='density')
    freq_1a, psd_1a = welch(ay_KBR1A[:, 1], 10.0, ('kaiser', 30.), ay_KBR1A[:, 0].__len__(), scaling='density')
    freq_kaiser, psd_kaiser = welch(ay_KBR1B_kaiser[1000: 860000, 0], 10.0, ('kaiser', 30.), ay_KBR1B_kaiser[1000: 860000, 0].__len__(), scaling='density')

    plt.style.use(['science', 'no-latex'])
    fig, ax = plt.subplots(figsize=(50, 25))
    ax.loglog(freq_1a, np.sqrt(psd_1a), linewidth=2, label=r'$KBR1A$')
    ax.loglog(freq_kaiser, np.sqrt(psd_kaiser), linewidth=2, label=r'$KBR1B-kaiser-10Hz$')
    ax.loglog(freq_crn, np.sqrt(psd_crn), linewidth=2, label=r'$KBR1B-JPL-0.2Hz$')
    ax.fill_between(freq_1a, np.sqrt(psd_1a), where=freq_1a <= 0.025, facecolor='xkcd:grey', alpha=0.3)
    ax.tick_params(labelsize=25, width=2.9, length=15, which='major')
    ax.tick_params(which='minor', length=9)
    ax.set_xlabel('Frequency [Hz]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.set_xscale('log')
    ax.set_yscale('log')
    x_major = ticker.LogLocator(base=10.0, numticks=100)
    ax.xaxis.set_major_locator(x_major)
    locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_ylabel(r'$Range \, \, ASD [m/\sqrt{Hz}]$', fontsize=20)
    plt.legend(fontsize=15, loc='best')
    # plt.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    fig.savefig('..//images//kaiser-psd-comparison.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    main()
    # tof()
    # kaiser_compare()
