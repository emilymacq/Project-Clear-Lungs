

%% Load raw audio file
fileID = fopen('RECORD.RAW');
rec = fread(fileID,inf,'uint16');

% fix big values
idx = rec>(2.^15);
rec(idx) = rec(idx) - (2.^16);


% %% Load audio text file
% fileID = fopen('RECORD.txt','r');
% rec = fscanf(fileID,'%f');




% Extract data from each mic
n = 256;
m1 = []; m2 = []; m3 = []; m4 = [];
L = 4*n*floor(length(rec)/(4*n)); % find a length so that each mic has the same length recording
for i = 1:(n*4):L
    m1 = [m1; rec((i+0*n):(i+1*n-1))];
    m2 = [m2; rec((i+1*n):(i+2*n-1))];
    m3 = [m3; rec((i+2*n):(i+3*n-1))];
    m4 = [m4; rec((i+3*n):(i+4*n-1))];
end
t = (0:length(m1)-1) ./ 44100;


disp('done');
% %%
% figure; hold on;
% plot(m2);
% xlim([60100 88000]); ylim([-2000 2000]);
% title('Mic 2');
% xlabel('t (sec)');



%% Plot each Mic

xrange = [0,5];
yrange = [-1000,1000];

figure; hold on;
subplot(221); plot(t,m1);
xlim(xrange); ylim(yrange);
title('Mic 1');
xlabel('t (sec)');

subplot(222); plot(t,m2);
xlim(xrange); ylim(yrange);
title('Mic 2');
xlabel('t (sec)');

subplot(223); plot(t,m3);
xlim(xrange); ylim(yrange);
title('Mic 3');
xlabel('t (sec)');

subplot(224); plot(t,m4);
xlim(xrange); ylim(yrange);
title('Mic 4');
xlabel('t (sec)');

%% filter 60hz
Fs = 44100;
d = designfilt('bandstopiir','FilterOrder',2, ...
               'HalfPowerFrequency1',59,'HalfPowerFrequency2',61, ...
               'DesignMethod','butter','SampleRate',Fs);
m2_60 = filtfilt(d,m2);

%% Listen to a mic



soundsc(m2,44100);


%% Find Frequency Spectrum
Fs = 44100;
L = length(m1);
f = Fs/2*linspace(0,1,L/2+1);

Y1 = fft(m1);
Y2 = fft(m2);
Y3 = fft(m3);
Y4 = fft(m4);

% Plot single-sided amplitude spectrum.
xrange = [0, 1000];
yrange = [0, max(abs([Y1;Y2;Y3;Y4]))];
figure; hold on;

subplot(221); plot(f,2*abs(Y1(1:L/2+1)));
xlim(xrange); ylim(yrange);
title('Mic 1');
xlabel('F (Hz)');

subplot(222); plot(f,2*abs(Y2(1:L/2+1)), 'LineWidth',2);
xlim(xrange); ylim(yrange);
title('Mic 2');
xlabel('F (Hz)');

subplot(223); plot(f,2*abs(Y3(1:L/2+1)));
xlim(xrange); ylim(yrange);
title('Mic 3');
xlabel('F (Hz)');

subplot(224); plot(f,2*abs(Y4(1:L/2+1)));
xlim(xrange); ylim(yrange);
title('Mic 4');
xlabel('F (Hz)');








