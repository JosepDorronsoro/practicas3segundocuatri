% P2 Ejercicio 1a

clear all;
close all;
clc;

% numerador:

hn1=[1 -exp(pi/3*1i)];
hn2=[1 -exp(2*pi/3*1i)];
hn3=[1 -exp(-pi/3*1i)];
hn4=[1 -exp(-2*pi/3*1i)];

% denominador:

hd1=[1 -0.95*exp(pi/3*1i)];
hd2=[1 -0.95*exp(2*pi/3*1i)];
hd3=[1 -0.95*exp(-pi/3*1i)];
hd4=[1 -0.95*exp(-2*pi/3*1i)];

% recordar usar convolución, no el producto de vectores:

h1=conv(hn1, conv(hn2, conv(hn3, hn4))); % numerador 
h2=conv(hd1, conv(hd2, conv(hd3, hd4))); % denominador

% obtenemos la transformada H y su frecuencia en radianes:
[H, w] = freqz(h1, h2, 1024);

% graficaremos la magnitud, la fase y el retardo de grupo 
% del filtro:

figure; 
subplot(3, 1, 1);
plot(w/pi, 20*log10(abs(H)));
xlabel('Frecuencia normalizada (\times\pi rad/sample)');
ylabel('Magnitud (dB)');
subplot(3, 1, 2);
plot(w/pi, angle(H)*(180/pi));
xlabel('Frecuencia normalizada (\times\pi rad/sample)');
ylabel('Módulo (en grados)');
[gd, ] = grpdelay(h1, h2, 1024);
subplot(3, 1, 3);
plot(w/pi, gd);
xlabel('Frecuencia normalizada (\times\pi rad/sample)');
ylabel('Retardo de grupo (en muestras)');