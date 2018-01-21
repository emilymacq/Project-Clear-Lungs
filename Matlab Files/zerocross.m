[x Fs] = audioread('Crackles - Early Inspiratory (Rales).mp3');
left_channel = x(:,1);

zero_cross = size(find(diff(left_channel>0)~=0)+1)