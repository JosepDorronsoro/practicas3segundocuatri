% Ejercicio 1 

clear all;
close all;
clc;

% CARGAMOS LA SEÑAL

load t1d

figure;
stem(t1d)
title('t1d')

% La señal original se puede visualizar
% sabiendo que fue muestreada a 8kHz

fs=8000;
tr=t1d;
Ns=length(tr); % no. muestras de trama
eje_t=(0:Ns-1)/8; % eje temporal en ms
figure, plot(eje_t,tr)
xlabel ('t (ms.)')
ylabel ('x[n]')
title('Forma de onda temporal de trama de voz')

% DEFINIMOS LA VENTANA RECTANGULAR

N_fft=512;
lw=30; %longitud de la ventana
v_rect=[ones(lw,1) 
zeros(N_fft-lw,1)];
xw=v_rect;

XW=fft(xw,N_fft);
W=(0:N_fft-1)*(2*pi/N_fft);

figure,plot(W,20*log10(abs(XW)))
xlabel ('w (rad)')
ylabel ('dB')
title('Espectro de una ventana rectangular de 30 puntos')
grid

% DEFINIMOS LA VENTANA HAMMING

figure;
hammingWindow = hamming(lw);
XH = fft(hammingWindow, N_fft);
plot(W, 20*log10(XH));
xlabel ('w (rad)')
ylabel ('dB')
title('Espectro de una ventana de Hamming de 30 puntos')
grid

% ESPECTRO DE LAS PRIMERAS 30 MUESTRAS DE VOZ -  RECTANGULAR
N_fft=512;
xw=tr(1:30);
XW=fft(xw,N_fft);
W=(0:N_fft-1)*(2*pi/N_fft);
eje_f=W*fs/(2*pi);
figure,plot(eje_f(1:N_fft/2),20*log10(abs(XW(1:N_fft/2))))
xlabel ('f (Hz)')
ylabel ('dB')
title('Espectro de las primeras 30 muestras de la trama de voz')
grid

% ESPECTRO DE LAS PRIMERAS 30 MUESTRAS DE VOZ -  HAMMING
N_fft=512;
hammingWindow = hamming(lw);
xw = tr(1:30) .* hammingWindow;
XH = fft(xw, N_fft); 
W=(0:N_fft-1)*(2*pi/N_fft);
eje_f=W*fs/(2*pi);
figure,plot(eje_f(1:N_fft/2),20*log10(abs(XH(1:N_fft/2))))
xlabel ('f (Hz)')
ylabel ('dB')
title('Espectro de las primeras 30 muestras de la trama de voz')
grid

% ESPECTRO DE LAS PRIMERAS 30 MUESTRAS DE VOZ -  RECTANGULAR - 2048
N_fft=2048;
xw=tr(1:240);
XW=fft(xw,N_fft);
W=(0:N_fft-1)*(2*pi/N_fft);
eje_f=W*fs/(2*pi);
figure,plot(eje_f(1:N_fft/2),20*log10(abs(XW(1:N_fft/2))))
xlabel ('f (Hz)')
ylabel ('dB')
title('Espectro de la trama de voz enventanada con N_{ftt}=2048')
grid

% ESPECTRO DE LAS PRIMERAS 30 MUESTRAS DE VOZ -  HAMMING - 2048
N_fft=2048;
hammingWindow = hamming(240);
xw = tr(1:240) .* hammingWindow(1:240);
XH = fft(xw, N_fft); 
W=(0:N_fft-1)*(2*pi/N_fft);
eje_f=W*fs/(2*pi);
figure,plot(eje_f(1:N_fft/2),20*log10(abs(XH(1:N_fft/2))))
xlabel ('f (Hz)')
ylabel ('dB')
title('Espectro de las primeras 30 muestras de la trama de voz')
grid