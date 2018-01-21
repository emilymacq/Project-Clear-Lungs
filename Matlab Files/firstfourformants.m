[x Fs] = audioread('Crackles - Early Inspiratory (Rales).mp3');
left_channel = x(:,1);
fleft_channel = real(fft(left_channel));
plot(fleft_channel);
[sorted_fx, sorted_indices] = sort(fleft_channel);
first_four = sorted_indices(end-4:end)