% Ejercicio  1

clear all;
close all;
clc;

addpath('./P6images&code/')

% Cargamos la imagen

[ima, map] = imread('edificio_bw_512.bmp');

% La pasamos a escala de grises:

ima = ind2gray(ima, map);

% Ahora ejecutamos el algoritmo de Canny sin argumentos:

E = edge(ima, 'canny');

% Y graficamos los contornos en blanco y cyan

canny(:, :, 1) = ima;                   % R
canny(:, :, 2) = ima + uint8(E*255);    % G
canny(:, :, 3) = ima + uint8(E*255);    % B

figure;
imshow(canny)

% Ahora lo repetimos para distintos valores del umbral de Canny: 

figure;
k=1;
for th=0.2:0.1:0.5

subplot(1, 4, k);
E = edge(ima, 'canny', th);
canny(:, :, 1) = ima;                   % R
canny(:, :, 2) = ima + uint8(E*255);    % G
canny(:, :, 3) = ima + uint8(E*255);    % B
imshow(canny)
title(sprintf('Umbral: %.1f', th))
k=k+1;

end % for th

figure;
k=1;
for th=0.2:0.1:0.5
for sig=0.1:0.1:0.4

subplot(4, 4, k);
E = edge(ima, 'canny', th, sig);
canny(:, :, 1) = ima;                   % R
canny(:, :, 2) = ima + uint8(E*255);    % G
canny(:, :, 3) = ima + uint8(E*255);    % B
imshow(canny)
title(sprintf('Umbral: %.1f, Sigma. %.1f', th, sig))
k=k+1;

end % for sig
end % for th