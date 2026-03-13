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

% DEFINIMOS LA VENTANA

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

