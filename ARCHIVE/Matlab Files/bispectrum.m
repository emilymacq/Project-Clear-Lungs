
[Y, FS] = audioread('Wheeze.mp3');
plot(Y(:,1));
F = ifft(Y);
G = fft(Y);
Bi = F(95)*G(90)*G(5);