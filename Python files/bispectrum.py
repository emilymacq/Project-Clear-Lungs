"""https://github.com/synergetics/spectrum#cross-bispectrum-direct"""

# DRAFT by Dana Zhu
# Still need to do getBGS --> calculate ratio of area under curve for each frequency

import sys
import scipy
from scipy import io
from scipy.io import wavfile

from spectrum import bispectrumd
from spectrum import cumest

# EFFECT: calculate Bispectrum
# output 1) Bspec : estimated bispectrum: an nfft x nfft array, origin at center, axes down and right
# output 2) waxis : vector of frequencies associated with rows and cols of Bspec
def getBispectrum():
	# I used the function under Bispectrum Direct (using fft)
	bispectrumd(y, nfft=None, wind=None, nsamp=None, overlap=None)
	
	
# calculate BGS by computing ratio of area under the curve
def getBGS(Bspec, waxis):
	# frequency 90 Hz

	# ferquency 5 kHz

	# frequency 6 kHz

	# frequency 10.5 kHz



def main():
    file = sys.argv[1]
    print(file)
    sampling_rate, sound = scipy.io.wavfile.read(file)
    getBGS(getBispectrum)


if __name__ == '__main__':
    main()
