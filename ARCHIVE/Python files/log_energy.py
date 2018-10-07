""" https://github.com/tyiannak/pyAudioAnalysis/blob/master/audioFeatureExtraction.py"""
import sys
import numpy
import scipy
import math
from math import log
from numpy import sum, float64
from scipy import io
from scipy.io import wavfile

def logEnergy(sound):
    """Computes signal energy"""
    signalEnergy = numpy.sum(sound ** 2) / numpy.float64(len(sound))
    return log(signalEnergy)

def main():
	file = sys.argv[1]
	print(file)
	samplingRate, sound = scipy.io.wavfile.read(file)
	logEnergy(sound)

if __name__ == '__main__':
	main()