% Ejercicio 2

clear all;
close all;
clc;

% aplicamos a la señal un filtro FIR

load t1d
fs=8000;
tr=t1d;
eje_n=0:49;% P = 50 (longitud respuesta impulsiva)
h=(0.99).^eje_n;
figure,stem(eje_n,h)
y_cl=conv(tr,h); % Longitud conv. lineal: L+P-1 = 240 + 50 -1 = 289
figure,subplot(311),stem(tr),
title('Señal temporal de entrada, x[n]'),
subplot(312),stem(h),title('Respuesta impulsiva del filtro, h[n]'),
subplot(313),stem(y_cl),title('Salida y[n], como convolución lineal'),

% ahora lo hacemos en el dominio de la frecuencia con N_fft=256

N_fft = 256;
H = fft(h, N_fft);
X = fft(tr, N_fft);
Y = H(:) .* X(:); 
y = ifft(Y, N_fft);
w = linspace(-pi, pi, N_fft);

% esto es lo que está pasando en frecuencia: 
figure;
subplot(311), plot(w, fftshift(20*log10(abs(X)))), title('X(e^{jw})'), axis tight,
subplot(312), plot(w, fftshift(20*log10(abs(H)))), title('H(e^{jw})'), axis tight,
subplot(313), plot(w, fftshift(20*log10(abs(Y)))), title('Y(e^{jw})'), axis tight;

% y esto es lo que pasa en tiempo:

y_cl(257:end)=[];

figure; 
subplot(311), stem(y_cl), title('y[n] - convolucion'),
subplot(312), stem(y), title('y[n] - producto en frecuencia'),
subplot(313), stem(abs(y_cl-y)), title('y[n]_c - y[n]_f');



% ahora lo hacemos en el dominio de la frecuencia con N_fft=512

N_fft = 512;
H = fft(h, N_fft);
X = fft(tr, N_fft);
Y = H(:) .* X(:); 
y = ifft(Y, N_fft);
w = linspace(-pi, pi, N_fft);

% esto es lo que está pasando en frecuencia: 
figure;
subplot(311), plot(w, fftshift(20*log10(abs(X)))), title('X(e^{jw})'), axis tight,
subplot(312), plot(w, fftshift(20*log10(abs(H)))), title('H(e^{jw})'), axis tight,
subplot(313), plot(w, fftshift(20*log10(abs(Y)))), title('Y(e^{jw})'), axis tight;

% y esto es lo que pasa en tiempo:

y(257:end)=[];

figure; 
subplot(311), stem(y_cl), title('y[n] - convolucion'),
subplot(312), stem(y), title('y[n] - producto en frecuencia'),
subplot(313), stem(abs(y_cl-y)), title('y[n]_c - y[n]_f');