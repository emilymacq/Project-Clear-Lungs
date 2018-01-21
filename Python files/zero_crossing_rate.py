# https://github.com/tyiannak/pyAudioAnalysis/blob/master/audioFeatureExtraction.py
import sys
import numpy
import scipy
from numpy import sum, abs, diff, sign, float64
from scipy import io
from scipy.io import wavfile
import urllib
import pydub

def compute_ZCR(sound):
    """Computes zero crossing rate"""
    count = len(sound)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(sound)))) / 2
    zcr = (numpy.float64(countZ) / numpy.float64(count-1.0))
    print(zcr)

def main():
	file = sys.argv[1]
	print(file)
	samplingRate, sound = scipy.io.wavfile.read(file)
	compute_ZCR(sound)

if __name__ == '__main__':
	main()