"""https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html#scipy.stats.kurtosis"""

import scipy
import sys
from scipy import stats
from scipy.stats import kurtosis

def computeKurtosis(sound):
    """Computes kurtosis"""
    kurtosis_value = kurtosis(sound)
    return kurtosis_value

def main():
	file = sys.argv[1]
	print(file)
	samplingRate, sound = scipy.io.wavfile.read(file)
	computeKurtosis(sound)

if __name__ == '__main__':
	main()