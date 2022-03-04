# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, kaiserord, firwin, filtfilt
import glob
import dask.dataframe as dd
import platform
from scipy.stats import pearsonr, norm
from matplotlib import font_manager


def brush_1_0(xy):
    arr_1 = xy[xy[:, 1] == 1]


def iono_corr():
    # cutoff frequency (Hz)
    fcut = 0.035
    fontP = font_manager.FontProperties()
    fontP.set_family('SimHei')
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # read in data
    dd_kbr1b = dd.read_csv(
        urlpath='..//..//..//gracefo_dataset//gracefo_1B_2018-12-22_RL04.ascii.noLRI//KBR1B_2018-12-22_Y_04.txt',
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
    shadow_c = np.loadtxt(
        fname='..//..//..//gracefo_dataset//gracefo_1A_2018-12-22_RL04.ascii.noLRI//SHA1A_2018-12-22_C_04.txt')
    shadow_d = np.loadtxt(
        fname='..//..//..//gracefo_dataset//gracefo_1A_2018-12-22_RL04.ascii.noLRI//SHA1A_2018-12-22_D_04.txt')

    dd_kbr1b_1 = dd.read_csv(
        urlpath='..//..//..//gracefo_dataset//gracefo_1B_2019-06-22_RL04.ascii.noLRI//KBR1B_2019-06-22_Y_04.txt',
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
    shadow_c_1 = np.loadtxt(
        fname='..//..//..//gracefo_dataset//gracefo_1A_2019-06-22_RL04.ascii.noLRI//SHA1A_2019-06-22_C_04.txt')
    shadow_d_1 = np.loadtxt(
        fname='..//..//..//gracefo_dataset//gracefo_1A_2019-06-22_RL04.ascii.noLRI//SHA1A_2019-06-22_D_04.txt')

    iono_corr_hf = dd_kbr1b['iono_corr'].compute().to_numpy() - kaiser(dd_kbr1b['iono_corr'].compute().to_numpy(), 0.1, fcut)
    iono_corr_hf = -(iono_corr_hf * 32e9**2) / 40.3

    iono_corr_hf_1 = dd_kbr1b_1['iono_corr'].compute().to_numpy() - kaiser(dd_kbr1b_1['iono_corr'].compute().to_numpy(), 0.1, fcut)
    iono_corr_hf_1 = -(iono_corr_hf_1 * 32e9**2) / 40.3

    plt.style.use(['science', 'no-latex', 'high-vis'])
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    time_span = np.linspace(0, 86400, iono_corr_hf.__len__())
    ax[0].plot(time_span, iono_corr_hf, linewidth=1)
    ax[0].fill_between(shadow_c[:, 0] - shadow_c[0, 0], 0, 1, where=shadow_c[:, 1] == 0,
                    color='grey', alpha=0.5, transform=ax[0].get_xaxis_transform())
    ax[0].fill_between(shadow_d[:, 0] - shadow_d[0, 0], 0, 1, where=shadow_d[:, 1] == 0,
                    color='grey', alpha=0.5, transform=ax[0].get_xaxis_transform())
    ax[0].set_ylim([-0.4e15, 0.4e15])
    ax[0].set_xlim([500, 85500])
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(-5, 2))
    ax[0].set_xlabel(u"自2018年12月22日00:00:00开始GPS时 [s]", fontsize=20, fontproperties=fontP)
    ax[0].set_ylabel(r'0.04-0.08 Hz频段的水平电子总数 [TECU]', fontsize=20, fontproperties=fontP)
    ax[0].yaxis.get_offset_text().set_fontsize(20)
    ax[0].spines['top'].set_linewidth(2)
    ax[0].spines['bottom'].set_linewidth(2)
    ax[0].spines['left'].set_linewidth(2)
    ax[0].spines['right'].set_linewidth(2)
    ax[0].text(5000, -3.5e14, "(a)", fontsize=15)
    # ax[0].legend(fontsize=20, loc='best', frameon=False)
    ax[0].tick_params(labelsize=25, width=2.9)

    ax[1].plot(time_span, iono_corr_hf_1, linewidth=1)
    ax[1].fill_between(shadow_c_1[:, 0] - shadow_c_1[0, 0], 0, 1, where=shadow_c_1[:, 1] == 0,
                    color='grey', alpha=0.5, transform=ax[1].get_xaxis_transform())
    ax[1].fill_between(shadow_d_1[:, 0] - shadow_d_1[0, 0], 0, 1, where=shadow_d_1[:, 1] == 0,
                    color='grey', alpha=0.5, transform=ax[1].get_xaxis_transform())
    ax[1].set_ylim([-3e14, 3e14])
    ax[1].set_xlim([500, 85500])
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(-5, 2))
    ax[1].set_xlabel(u"自2019年6月22日00:00:00开始GPS时 [s]", fontsize=20, fontproperties=fontP)
    ax[1].set_ylabel(r'0.04-0.08 Hz频段的水平电子总数 [TECU]', fontsize=20, fontproperties=fontP)
    ax[1].yaxis.get_offset_text().set_fontsize(20)
    ax[1].spines['top'].set_linewidth(2)
    ax[1].spines['bottom'].set_linewidth(2)
    ax[1].spines['left'].set_linewidth(2)
    ax[1].spines['right'].set_linewidth(2)
    ax[1].text(3000, -2.5e14, "(b)", fontsize=15)
    # ax[1].legend(fontsize=20, loc='best', frameon=False)
    ax[1].tick_params(labelsize=25, width=2.9)
    plt.tight_layout()
    plt.show()
    fig.savefig("..//images//TEC_s_w.png", dpi=300)

    # axins1 = ax.inset_axes((0.55, 0.75, 0.2, 0.2))
    # axins1.tick_params(labelsize=15, width=1)
    # axins1.yaxis.get_offset_text().set_fontsize(14)
    # axins1.spines['top'].set_linewidth(2)
    # axins1.spines['bottom'].set_linewidth(2)
    # axins1.spines['left'].set_linewidth(2)
    # axins1.spines['right'].set_linewidth(2)
    # zone_left = 14876
    # zone_right = 14975
    # x_ratio = 0  # x轴显示范围的扩展比例
    # y_ratio = 0.05  # y轴显示范围的扩展比例
    # xlim0 = time_span[zone_left] - (time_span[zone_right] - time_span[zone_left]) * x_ratio
    # xlim1 = time_span[zone_right] + (time_span[zone_right] - time_span[zone_left]) * x_ratio
    # y = np.hstack((iono_corr_hf[zone_left: zone_right]))
    # ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
    # ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio
    # axins1.set_xlim(xlim0, xlim1)
    # axins1.set_ylim(ylim0, ylim1)
    # tx0 = xlim0
    # tx1 = xlim1
    # ty0 = ylim0
    # ty1 = ylim1
    # sx = [tx0,tx1,tx1,tx0,tx0]
    # sy = [ty0,ty0,ty1,ty1,ty0]
    # ax.plot(sx,sy,"black")
    # mark_inset(ax, axins1, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    # axins1.plot(time_span[zone_left: zone_right], iono_corr_hf[zone_left: zone_right])
# 
    # axins2 = ax.inset_axes((0.25, 0.75, 0.2, 0.2))
    # axins2.tick_params(labelsize=15, width=1)
    # axins2.yaxis.get_offset_text().set_fontsize(14)
    # axins2.spines['top'].set_linewidth(2)
    # axins2.spines['bottom'].set_linewidth(2)
    # axins2.spines['left'].set_linewidth(2)
    # axins2.spines['right'].set_linewidth(2)
    # zone_left = 7000
    # zone_right = 7250
    # x_ratio = 0  # x轴显示范围的扩展比例
    # y_ratio = 0.05  # y轴显示范围的扩展比例
    # xlim0 = time_span[zone_left] - (time_span[zone_right] - time_span[zone_left]) * x_ratio
    # xlim1 = time_span[zone_right] + (time_span[zone_right] - time_span[zone_left]) * x_ratio
    # y = np.hstack((iono_corr_hf[zone_left: zone_right]))
    # ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
    # ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio
    # axins2.set_xlim(xlim0, xlim1)
    # axins2.set_ylim(ylim0, ylim1)
    # tx0 = xlim0
    # tx1 = xlim1
    # ty0 = ylim0
    # ty1 = ylim1
    # sx = [tx0,tx1,tx1,tx0,tx0]
    # sy = [ty0,ty0,ty1,ty1,ty0]
    # ax.plot(sx,sy,"black")
    # mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec='k', lw=1)
    # axins2.plot(time_span[zone_left: zone_right], iono_corr_hf[zone_left: zone_right])
# 
    # plt.show()

    freq_k1ac, psd_k1ac = welch(iono_corr_hf[500: 85500],
                                10., ('kaiser', 30.),
                                iono_corr_hf[500: 85500].__len__(),
                                scaling='density')

    # ax[1].stem(thr_c['rcvtime_intg'].compute().to_numpy()[np.nonzero(thr_c['on_time_att_ctrl_2_2'].compute().to_numpy())] - thr_c['rcvtime_intg'].compute().to_numpy()[0],
    #            thr_c['on_time_att_ctrl_2_2'].compute().to_numpy()[np.nonzero(thr_c['on_time_att_ctrl_2_2'].compute().to_numpy())],
    #            label='att_2_2', linefmt='k-', markerfmt=' ', basefmt='k.')
    # ax[1].stem(thr_c['rcvtime_intg'].compute().to_numpy()[np.nonzero(thr_c['on_time_att_ctrl_1_6'].compute().to_numpy())] - thr_c['rcvtime_intg'].compute().to_numpy()[0],
    #            thr_c['on_time_att_ctrl_1_6'].compute().to_numpy()[np.nonzero(thr_c['on_time_att_ctrl_1_6'].compute().to_numpy())],
    #            label='att_1_6', linefmt='y-', markerfmt=' ', basefmt='y.')
    # ax[1].set_xlim([500, 85500])
    # ax[1].set_ylabel(u'C星推力器 $[\mu s]$', fontsize=20, fontproperties=fontP)
    # ax[1].yaxis.get_offset_text().set_fontsize(24)
    # ax[1].legend(fontsize=20, loc='best', frameon=False)
    # ax[1].tick_params(labelsize=25, width=2.9)
    #
    # ax[2].stem(thr_d['rcvtime_intg'].compute().to_numpy()[np.nonzero(thr_d['on_time_att_ctrl_2_2'].compute().to_numpy())] - thr_d['rcvtime_intg'].compute().to_numpy()[0],
    #            thr_d['on_time_att_ctrl_2_2'].compute().to_numpy()[np.nonzero(thr_d['on_time_att_ctrl_2_2'].compute().to_numpy())],
    #            label='att_2_2', linefmt='k-', markerfmt=' ', basefmt='k.')
    # ax[2].stem(thr_d['rcvtime_intg'].compute().to_numpy()[np.nonzero(thr_d['on_time_att_ctrl_1_5'].compute().to_numpy())] - thr_d['rcvtime_intg'].compute().to_numpy()[0],
    #            thr_d['on_time_att_ctrl_1_5'].compute().to_numpy()[np.nonzero(thr_d['on_time_att_ctrl_1_5'].compute().to_numpy())],
    #            label='att_1_5', linefmt='r-', markerfmt=' ', basefmt='r.')
    # ax[2].stem(thr_d['rcvtime_intg'].compute().to_numpy()[np.nonzero(thr_d['on_time_att_ctrl_1_2'].compute().to_numpy())] - thr_d['rcvtime_intg'].compute().to_numpy()[0],
    #            thr_d['on_time_att_ctrl_1_2'].compute().to_numpy()[np.nonzero(thr_d['on_time_att_ctrl_1_2'].compute().to_numpy())],
    #            label='att_1_2', linefmt='g-', markerfmt=' ', basefmt='g.')
    # ax[2].set_xlim([500, 85500])
    # ax[2].set_ylabel(u'D星推力器 $[\mu s]$', fontsize=20, fontproperties=fontP)
    # ax[2].yaxis.get_offset_text().set_fontsize(24)
    # ax[2].legend(fontsize=20, loc='best', frameon=False)
    # ax[2].tick_params(labelsize=25, width=2.9)

    # ax[1].plot(acc_c['rcvtime_intg'].compute().to_numpy() + acc_c['rcvtime_frac'].compute().to_numpy() * 1e-6 -
    #            acc_c['rcvtime_intg'].compute().to_numpy()[0],
    #            acc_c['lin_accl_x'].compute().to_numpy(),
    #            linewidth=1,
    #            label='lin_x')
    # ax[1].plot(acc_c['rcvtime_intg'].compute().to_numpy() + acc_c['rcvtime_frac'].compute().to_numpy() * 1e-6 -
    #            acc_c['rcvtime_intg'].compute().to_numpy()[0],
    #            acc_c['lin_accl_y'].compute().to_numpy(),
    #            linewidth=1,
    #            label='lin_y')
    # ax[1].plot(acc_c['rcvtime_intg'].compute().to_numpy() + acc_c['rcvtime_frac'].compute().to_numpy() * 1e-6 -
    #            acc_c['rcvtime_intg'].compute().to_numpy()[0],
    #            acc_c['lin_accl_z'].compute().to_numpy(),
    #            linewidth=1,
    #            label='lin_z')
    #
    # ax[1].set_ylabel(u'C星线性加速度 $[m/s^2]$', fontsize=20, fontproperties=fontP)
    # ax[1].yaxis.get_offset_text().set_fontsize(24)
    # ax[1].legend(fontsize=20, loc='best', frameon=False)
    # ax[1].tick_params(labelsize=25, width=2.9)
    #
    # ax[2].plot(acc_d['rcvtime_intg'].compute().to_numpy() + acc_d['rcvtime_frac'].compute().to_numpy() * 1e-6 -
    #            acc_d['rcvtime_intg'].compute().to_numpy()[0],
    #            acc_d['lin_accl_x'].compute().to_numpy(),
    #            linewidth=1,
    #            label='lin_x')
    # ax[2].plot(acc_d['rcvtime_intg'].compute().to_numpy() + acc_d['rcvtime_frac'].compute().to_numpy() * 1e-6 -
    #            acc_d['rcvtime_intg'].compute().to_numpy()[0],
    #            acc_d['lin_accl_y'].compute().to_numpy(),
    #            linewidth=1,
    #            label='lin_y')
    # ax[2].plot(acc_d['rcvtime_intg'].compute().to_numpy() + acc_d['rcvtime_frac'].compute().to_numpy() * 1e-6 -
    #            acc_d['rcvtime_intg'].compute().to_numpy()[0],
    #            acc_d['lin_accl_z'].compute().to_numpy(),
    #            linewidth=1,
    #            label='lin_z')
    #
    # ax[2].set_ylabel(u'D星线性加速度 [m/s^2]', fontsize=20, fontproperties=fontP)
    # ax[2].yaxis.get_offset_text().set_fontsize(24)
    # ax[2].legend(fontsize=20, loc='best', frameon=False)
    # ax[2].tick_params(labelsize=25, width=2.9)
    # ax[0].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)

    # fm._rebuild()
    # plt.style.use([
    #     'science',
    #     'no-latex',
    #     'high-vis',
    #     'cjk-sc-font'
    # ])

    # spec
    # fig, ax = plt.subplots(figsize=(50, 25))
    # pxx, freq, t, cax = ax.specgram(iono_corr_hf[2500: 861500],
    #                                 2000,
    #                                 Fs=10.0,
    #                                 noverlap=700,
    #                                 cmap=pylab.get_cmap('jet'))
    # ax.fill_between(shadow_c[:, 0] - shadow_c[0, 0],
    #                 0,
    #                 1,
    #                 where=shadow_c[:, 1] == 0,
    #                 color='black',
    #                 alpha=0.3,
    #                 transform=ax.get_xaxis_transform())
    # cbar = fig.colorbar(cax)
    # cbar.ax.tick_params(labelsize=25)
    # ax.set_xlabel(u"自2018年12月1日00:00:00开始GPS时 [s]", fontsize=20)
    # ax.set_ylabel(u"频率 [Hz]", fontsize=20)
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label1.set_fontproperties('stixgeneral')
    # ax.set_yscale('log')
    # ax.set_ylim(0.03, 1.0)
    # ax.set_xlim(0, 86400)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # #ax.legend(fontsize=20, loc='best', frameon=False)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)

    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.plot(dd_range - kaiser(dd_range, 5., fcut), linewidth=2)
    # ax.set_xlabel(r'$Sampling \,\, point$', fontsize=20)
    # ax.set_ylabel(r'$Double \,\, Differenced \,\, Range [m]$',
    #               fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.legend(fontsize=20, loc='best', frameon=False)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    #
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.set_xlabel(r'$Sampling \,\, point$', fontsize=20)
    # ax.set_ylabel(r'$Ionosphere \,\, correction (Ka \,\, band) [m]$', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.legend(fontsize=20, loc='best', frameon=False)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    #
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.loglog(freq_corr, np.sqrt(psd_corr), linewidth=2, label='Ka band')
    # ax.set_xlabel(r'$Frequency [Hz]$', fontsize=20)
    # ax.set_ylabel(r'$ASD [m/\sqrt{Hz}]$', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.legend(fontsize=20, loc='best', frameon=False)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    #
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.loglog(freq_ddra, np.sqrt(psd_ddra), linewidth=2, label='double differenced range')
    # ax.semilogx(np.linspace(0.0001, 5, 10000),
    #             2.62 *
    #             np.sqrt(1 + (0.003 / np.linspace(0.0001, 5, 10000)**2)) * 1e-6,
    #             label='stochastic error requirement')
    # ax.set_xlabel(r'$Frequency [Hz]$', fontsize=20)
    # ax.set_ylabel(r'$ASD [m/\sqrt{Hz}]$', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.legend(fontsize=20, loc='best', frameon=False)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    # plt.show()

    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.hist(iono_corr_hf,
    #         bins=np.linspace(-0.0001, 0.0001, 100),
    #         label='Ka band')
    # ax.plot(np.linspace(-0.0001, 0.0001, 100),
    #         norm.pdf(np.linspace(-0.0001, 0.0001, 100).astype(float), mu.astype(float), std.astype(float)), 'k', linewidth=2)
    # ax.set_xlabel(r'$Sampling \,\, points$', fontsize=20)
    # ax.set_ylabel(r'$Ionosphere \,\, correction \,\, residual [m]$',
    #               fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.legend(fontsize=20, loc='best', frameon=False)
    # ax.tick_params(labelsize=25, width=2.9)
    # # ax.set_title(f"Fit result for cutoff frequenct {fcut}: mu={mu}, std={std}")
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    #
    # freq_stoc, psd_stoc = welch(iono_corr_hf,
    #                             10., ('kaiser', 30.),
    #                             iono_corr.__len__(),
    #                             scaling='density')
    # freq_1b, psd_1b = welch(iono_kbr1b - kaiser(iono_kbr1b, 0.1, 0.02),
    #                         0.2, ('kaiser', 30.),
    #                         iono_kbr1b.__len__(),
    #                         scaling='density')
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.loglog(freq_stoc, np.sqrt(psd_stoc), linewidth=2, label='Ka band')
    # ax.loglog(freq_1b, np.sqrt(psd_1b), linewidth=2, label='1b')
    # ax.semilogx(np.linspace(0.0001, 5, 10000), 2.62 * np.sqrt(1 + (0.003 / np.linspace(0.0001, 5, 10000) ** 2)) * 1e-6, label='stochastic error requirement')
    # ax.set_xlabel(r'$Frequency [Hz]$', fontsize=20)
    # ax.set_ylabel(r'$ASD [m/\sqrt{Hz}]$', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.legend(fontsize=20, loc='best', frameon=False)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)



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
    fcut = 0.004  # Hz
    fs_1b = 0.2  # Hz
    fs_1a = 10  # Hz

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

    for index, (c_1a, d_1a, b, uso_c, uso_d) in enumerate(
            zip(kbr1a_c_filename, kbr1a_d_filename, kbr1b_y_filename, uso1b_c_filename, uso1b_d_filename)):
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
                                     'ka_freq': np.longdouble}, )
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
                                 }, )

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
            (1. / (2. * np.pi * kbr1a_c_k_snr[index])) ** 2 +
            (1. / (2. * np.pi * kbr1a_d_k_snr[index])) ** 2)
        phase_ka = c / (freq_ka_c + freq_ka_d) * np.sqrt(
            (1. / (2. * np.pi * kbr1a_c_ka_snr[index])) ** 2 +
            (1. / (2. * np.pi * kbr1a_d_ka_snr[index])) ** 2)

        kbr1a_sn[index] = np.sqrt(((freq_k_e ** 2 * phase_k) ** 2 + (freq_ka_e ** 2 * phase_ka) ** 2) /
                                  (freq_k_e ** 2 - freq_ka_e ** 2) ** 2) * np.sqrt(1. / 5.)

        # low-pass filter ionosphere correction
        iono_corr[index] = np.mean(
            dd_kbr1b.iono_corr.compute().to_numpy() - kaiser(dd_kbr1b.iono_corr.compute().to_numpy(), 0.1, 0.02))

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


def kaiser(x, fq, cutoff_hz, ripple_db=600.):
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


def outliers_reconstruction(iarray, threshold):
    """ This is a function that detect the outliers of certain time series exceeding the threshold
        and then, reconstruct them using Lagrange interpolation.

        Input variables:
            iarray: The input time series, whose size should be (N, 2), which will be checked in the
                    function. The first column contains the time stamp or time index, and the other
                    contains the physical variable.
            threshold: if the magnitude of certain physical quantity exceeds the threshold, it will
                        be marked as an outlier, and will be replaced by the interpolated value using
                        Lagrange interpolation

        Output variable:
            oarray: The first column of the output array is the same as the first column of the input
                    array, while the only difference of the second column between the two arrays is
                    the outliers in the input array are replaced by the interpolated value
    """

    # check if the size of iarray is valid
    if np.shape(iarray)[1] != 2:
        raise ValueError("The size of the input array for the function outliers_reconstruction is wrong")

    # detect the outliers
    nfloat = np.ceil(iarray.__len__() / 10000.0)
    outliers_index = np.asarray([], dtype=np.int64)
    for _, n in enumerate(np.arange(nfloat)):
        n = np.int64(n)
        p_residual = np.abs(
            iarray[n * 10000: (n + 1) * 10000, 1] - np.zeros(iarray[n * 10000: (n + 1) * 10000, 1].__len__()))
        outliers_index = np.append(outliers_index,
                                   np.where(p_residual > threshold * np.std(iarray[n * 10000: (n + 1) * 10000, 1]))[
                                       0] + n * 10000)
    return outliers_index


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
    np.set_printoptions(precision=3)
    iono_corr()
    # outlier()
    # system_noise_iono_residual()
