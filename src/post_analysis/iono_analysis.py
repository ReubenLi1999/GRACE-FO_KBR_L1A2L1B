import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
from matplotlib import font_manager
from scipy.signal import welch


def main():
    # matplotlib params
    fontP = font_manager.FontProperties()
    fontP.set_family('SimHei')
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # load ionosphere correction
    iono_corr_2018_12_01 = np.loadtxt(
        fname='..//..//..//..//gracefo_dataset//gracefo_1B_2018-12-01_RL04.ascii.noLRI//KBR1B_2018-12-01_Y_04.txt',
        skiprows=162,
        usecols=(4,)
    )
    iono_corr_2019_01_01 = np.loadtxt(
        fname='..//..//..//..//gracefo_dataset//gracefo_1B_2019-01-01_RL04.ascii.noLRI//KBR1B_2019-01-01_Y_04.txt',
        skiprows=162,
        usecols=(4,)
    )
    iono_corr_2019_02_01 = np.loadtxt(
        fname='..//..//..//..//gracefo_dataset//gracefo_1B_2019-02-01_RL04.ascii.noLRI//KBR1B_2019-02-01_Y_04.txt',
        skiprows=162,
        usecols=(4,)
    )
    iono_corr_2019_03_01 = np.loadtxt(
        fname='..//..//..//..//gracefo_dataset//gracefo_1B_2019-03-01_RL04.ascii.noLRI//KBR1B_2019-03-01_Y_04.txt',
        skiprows=162,
        usecols=(4,)
    )
    iono_corr_2019_04_01 = np.loadtxt(
        fname='..//..//..//..//gracefo_dataset//gracefo_1B_2019-04-01_RL04.ascii.noLRI//KBR1B_2019-04-01_Y_04.txt',
        skiprows=162,
        usecols=(4,)
    )
    iono_corr_2019_05_01 = np.loadtxt(
        fname='..//..//..//..//gracefo_dataset//gracefo_1B_2019-05-01_RL04.ascii.noLRI//KBR1B_2019-05-01_Y_04.txt',
        skiprows=162,
        usecols=(4,)
    )


    freq_2018_12_01, psd_2018_12_01 = welch(
        iono_corr_2018_12_01,
        0.2, ('kaiser', 15.),
        iono_corr_2018_12_01.__len__() / 5,
        scaling='density')
    freq_2019_01_01, psd_2019_01_01 = welch(
        iono_corr_2019_01_01,
        0.2, ('kaiser', 15.),
        iono_corr_2019_01_01.__len__() / 5,
        scaling='density')
    freq_2019_02_01, psd_2019_02_01 = welch(
        iono_corr_2019_02_01,
        0.2, ('kaiser', 15.),
        iono_corr_2019_02_01.__len__() / 5,
        scaling='density')
    freq_2019_03_01, psd_2019_03_01 = welch(
        iono_corr_2019_03_01,
        0.2, ('kaiser', 15.),
        iono_corr_2019_03_01.__len__() / 5,
        scaling='density')
    freq_2019_04_01, psd_2019_04_01 = welch(
        iono_corr_2019_04_01,
        0.2, ('kaiser', 15.),
        iono_corr_2019_04_01.__len__() / 5,
        scaling='density')
    freq_2019_05_01, psd_2019_05_01 = welch(
        iono_corr_2019_05_01,
        0.2, ('kaiser', 15.),
        iono_corr_2019_05_01.__len__() / 5,
        scaling='density')

    fig, ax = plt.subplots(5, 1, figsize=(20, 10))
    ax[0].loglog(freq_2018_12_01, np.sqrt(psd_2018_12_01), linewidth=2)
    ax[0].set_xticks([])
    ax[0].yaxis.get_offset_text().set_fontsize(24)
    ax[0].legend(fontsize=14, loc='best', frameon=False)
    ax[0].tick_params(labelsize=14, width=2.9)
    ax[0].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax[0].axvspan(4e-2, 8e-2, alpha=0.5, color='grey')
    ax[0].text(9e-4, 0.01, '2018-12-01',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax[0].transAxes,
        color='black', fontsize=15)

    ax[1].loglog(freq_2019_01_01, np.sqrt(psd_2019_01_01), linewidth=2)
    ax[1].set_xticks([])
    ax[1].yaxis.get_offset_text().set_fontsize(24)
    ax[1].legend(fontsize=14, loc='best', frameon=False)
    ax[1].tick_params(labelsize=14, width=2.9)
    ax[1].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax[1].axvspan(4e-2, 8e-2, alpha=0.5, color='grey')
    ax[1].text(9e-4, 0.01, '2019-01-01',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax[1].transAxes,
        color='black', fontsize=15)

    ax[2].loglog(freq_2019_03_01, np.sqrt(psd_2019_03_01), linewidth=2)
    ax[2].set_xticks([])
    ax[2].set_ylabel(r'$ASD [m/\sqrt{Hz}]$', fontsize=14)
    ax[2].yaxis.get_offset_text().set_fontsize(24)
    ax[2].legend(fontsize=14, loc='best', frameon=False)
    ax[2].tick_params(labelsize=14, width=2.9)
    ax[2].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax[2].axvspan(4e-2, 8e-2, alpha=0.5, color='grey')
    ax[2].text(9e-4, 0.01, '2019-03-01',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax[2].transAxes,
        color='black', fontsize=15)

    ax[3].loglog(freq_2019_04_01, np.sqrt(psd_2019_04_01), linewidth=2)
    ax[3].set_xticks([])
    ax[3].yaxis.get_offset_text().set_fontsize(24)
    ax[3].legend(fontsize=14, loc='best', frameon=False)
    ax[3].tick_params(labelsize=14, width=2.9)
    ax[3].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax[3].axvspan(4e-2, 8e-2, alpha=0.5, color='grey')
    ax[3].text(9e-4, 0.01, '2019-04-01',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax[3].transAxes,
        color='black', fontsize=15)

    ax[4].loglog(freq_2019_05_01, np.sqrt(psd_2019_05_01), linewidth=2)
    ax[4].set_xlabel('Frequency [Hz]', fontsize=14)
    ax[4].yaxis.get_offset_text().set_fontsize(24)
    ax[4].legend(fontsize=14, loc='best', frameon=False)
    ax[4].tick_params(labelsize=14, width=2.9)
    ax[4].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax[4].axvspan(4e-2, 8e-2, alpha=0.5, color='grey')
    ax[4].text(9e-4, 0.01, '2019-05-01',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax[4].transAxes,
        color='black', fontsize=15)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()



if __name__ == "__main__":
    main()