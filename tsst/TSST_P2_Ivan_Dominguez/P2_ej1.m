% P2 Ejercicio 1

clear all;
close all;
clc;

% cargamos la señal:

load("ecg_n1.mat");

% la visualizamos:

t=0:1/300:(3600/300)-(1/300);

figure;
plot(t,ecg_n1); axis tight;
title('ECG con ruido incluido de red de 50Hz'); xlabel('t(s)');

% visualizamos un zoom del primer segundo:

% la frecuencia de la señal son 50Hz:

% fijamos las muestras que hay en un segundo:
t_1_seg=0:1/300:1-(1/300);

figure;
plot(t_1_seg, ecg_n1(1:300)); axis tight; 
title('ECG (zoom de un segundo)'); xlabel('t(s)');

[pxx, w]=pwelch(ecg_n1);
figure;
plot(w, 10*log(pxx));
title('Esprectro de potencia de la señal de 0 a pi'); xlim([0, pi]);
xlabel('Frecuencia sin normalizar'); 
ylabel('$10log$(potencia)', Interpreter='latex');