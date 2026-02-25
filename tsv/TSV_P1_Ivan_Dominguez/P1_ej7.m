% Ejercicio 7

clear all;
close all;
clc;

% Cargamos la imagen de cameraman

[ima2_bw, map2_bw] = imread("cameraman.tif");

ima2_bw = im2double(ima2_bw);

% Creamos el enlace a la función mean:

func = @(x) mean(mean(x));

% Y ahora procesamos los bloques de 8x8, 
% transformando en un sólo píxel. 

[ima_procesada]= blkproc(ima2_bw,[8 8],func);

% Ahora representamos ambas imágenes en una figura.

figure; 
subplot(1, 2, 1);
imshow(ima2_bw);
title('Imagen original')
subplot(1, 2, 2);
imshow(ima_procesada)
title('Imagen procesada')
