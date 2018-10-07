import sys
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

def getMfcc(sound):
    (rate, sig) = wav.read(sound)
    mfcc_feat = mfcc(sig, rate)
    fbank_feat = logfbank(sig, rate)
    return mfcc_feat, fbank_feat

def main():
    # file = "Wheeze.wav"
    file = sys.argv[1]
    print(file)
    mfcc,fbank = getMfcc(file);
    print(fbank[1:3, :])

if __name__ == '__main__':
    main()



