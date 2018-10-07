[Y, FS] = audioread('Crackles - Low Pitched (Rales).mp3');
x = Y(:, 1);
maxval = max(x);
s = std(x);
crestFactor = 20*log10(maxval/s);