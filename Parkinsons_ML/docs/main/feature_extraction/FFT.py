
# import libraries
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

class fft_extractor:

    def calculate_fft(self, patient_data, col_name):
        sample_rate = len(patient_data)  # sampling rate
        sample_interval = 4  # sampling interval

        yf = fft(patient_data[col_name]) / sample_rate
        xf = np.linspace(0.0, sample_rate, sample_rate//2)
        print(len(yf), len(xf))

        ax = sns.lineplot(y=yf[0:sample_rate//2], x=xf)
        plt.show()
        return