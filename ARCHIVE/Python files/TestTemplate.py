import sys
import scipy
from scipy import io
from scipy.io import wavfile

def getVolume(sound):
    value = 0
    for sample in sound:
        value += abs(sample)
    print(value)

def main():
    file = sys.argv[1]
    print(file)
    sampling_rate, sound = scipy.io.wavfile.read(file)
    getVolume(sound)


if __name__ == '__main__':
    main()