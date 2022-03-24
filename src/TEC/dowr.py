import numpy as np
from scipy.signal import welch, kaiserord, firwin, filtfilt
import matplotlib.pyplot as plt


dowr = np.loadtxt("..//..//..//..//gracefo_dataset//gracefo_1A_2018-12-01_RL04.ascii.noLRI//DOWR1A_2018-12-01_Y_04.txt")

freq, psd = welch(
    dowr[:, 1], 10, ('kaiser', 30.), dowr[:, 0].__len__(), scaling='density')


plt.loglog(freq, np.sqrt(psd))
plt.show()