# Autogenerated with SMOP 0.35
from smop.core import *
# firstfourformants.m

    x,Fs=audioread('Crackles - Early Inspiratory (Rales).mp3',nargout=2)
# firstfourformants.m:1
    left_channel=x[:,1]
# firstfourformants.m:2
    fleft_channel=real(fft(left_channel))
# firstfourformants.m:3
    plot(fleft_channel)
    sorted_fx,sorted_indices=sort(fleft_channel,nargout=2)
# firstfourformants.m:5
    first_four=sorted_indices[end() - 4:end()]
# firstfourformants.m:6