import sys
import scipy
from scipy import io
from scipy.io import wavfile
import urllib
import pydub

def getVolume(sound):
    value = []
    for i in range(len(sound[0])):
        value.append(0)
    for sample in sound:
        for i in range(len(sample)):
            value[i] += abs(sample[i])
    print(value)

def main():
    file = sys.argv[1]
    print(file)
    #read mp3 file
    mp3 = pydub.AudioSegment.from_mp3(file)
    #convert to wav
    mp3.export("temp.wav", format="wav")
    sampling_rate, sound = scipy.io.wavfile.read("temp.wav")
    print(sampling_rate)
    print(sound[50570:50580])
    getVolume(sound)


if __name__ == '__main__':
    main()