[Y, FS] = audioread('Crackles - Low Pitched (Rales).mp3');
x = Y(:, 1);

mu = mean(x');
s = std(x);
% V is log variance
[M, V] = lognstat(mu, s);