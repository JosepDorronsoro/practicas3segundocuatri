% Ejercicio 1

clear all;
close all;
clc;

ima = imread('interferida.jpg');
ima = rgb2gray(ima);
ima = im2double(ima);

ancho = 1024; largo = 874;

v1 = [1/ancho, 0] ; v2 = [0, 1/largo];
x = 0:(1/ancho):1-(1/ancho);
y = 0:(1/largo):1-(0.1/largo);
[X, Y] = meshgrid(x, y);

A = 0.5;
f=20;
x1 = A * sin(2*pi*f*X);
x2 = A * cos(2*pi*f*X);

figure;
imshow(ima);
title('Imagen original')

% Tomamos la FFT

F = fft2(ima);
F_centrada = fftshift(F);

% La visualizamos

figure;
imshow(log(abs(F_centrada) + 1), []);
title('Espectro de Frecuencias (FFT)');

% La arreglamos

F_filtrada = F_centrada;

radio = 2; 
centro_fila = size(ima, 1)/2;
pico_izq = size(ima, 2)/2 - 20;
pico_der = size(ima, 2)/2 + 20;

F_filtrada(centro_fila-radio:centro_fila+radio, pico_izq-radio:pico_izq+radio) = 0;
F_filtrada(centro_fila-radio:centro_fila+radio, pico_der-radio:pico_der+radio) = 0;

% La vemos arreglada en frecuencia

figure;
imshow(log(abs(F_filtrada) + 1), []);
title('Espectro Filtrado (Picos eliminados)');

% La recuperamos arreglada

F_inversa = ifftshift(F_filtrada);
ima_recuperada = real(ifft2(F_inversa));

% La vemos arreglada

figure;
imshow(ima_recuperada, []);
title('Imagen Restaurada');