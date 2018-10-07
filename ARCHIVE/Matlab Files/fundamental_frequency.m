[x Fs] = audioread('Crackles - Early Inspiratory (Rales).mp3');
left_channel = x(:,1);

fleft_channel = fft(left_channel);
freq = 0:Fs/length(fleft_channel):Fs/2;
fleft_channel = fleft_channel(1:length(fleft_channel)/2+1);
plot(freq,abs(fleft_channel))
[maxval, index] = max(abs(fleft_channel));
freq(index)  %this is frequency corresponding to max value