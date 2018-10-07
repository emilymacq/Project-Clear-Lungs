[Y, FS] = audioread('Crackles - Low Pitched (Rales).mp3');
x = Y(:, 1);
plot(x);
skew = skewness(x);
kurt = kurtosis(x);