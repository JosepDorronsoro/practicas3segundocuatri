% Ejercicio 2.1

clear all;
close all;
clc;

% Definición de la matriz basada en la imagen (15x15)
imagen_matriz = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4;
    6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 5, 4, 4, 4, 4;
    7, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 4, 4, 4, 4;
    6, 6, 4, 3, 3, 4, 6, 6, 6, 6, 6, 4, 4, 4, 4;
    6, 4, 2, 1, 1, 2, 4, 6, 6, 6, 6, 5, 4, 4, 4;
    5, 3, 1, 0, 0, 1, 3, 6, 6, 6, 6, 6, 6, 5, 5;
    5, 3, 1, 0, 0, 1, 3, 5, 6, 6, 6, 7, 6, 6, 6;
    6, 4, 2, 1, 1, 2, 4, 6, 6, 6, 6, 6, 6, 6, 6;
    6, 6, 4, 3, 3, 4, 5, 6, 4, 3, 3, 4, 6, 6, 6;
    6, 6, 6, 5, 5, 6, 6, 4, 2, 1, 1, 2, 4, 6, 6;
    6, 6, 6, 6, 6, 6, 6, 3, 1, 0, 0, 1, 3, 5, 6;
    6, 6, 6, 6, 6, 6, 5, 3, 1, 0, 0, 1, 3, 5, 6;
    6, 6, 6, 6, 6, 6, 6, 4, 2, 1, 1, 2, 4, 6, 6;
    6, 6, 6, 6, 6, 6, 6, 5, 4, 3, 3, 4, 5, 6, 6;
    6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 6, 6, 6, 6
];

imagen_matriz = im2double(imagen_matriz);

% Para visualizarla como en la imagen de la izquierda (escala de grises)
% Usamos 'imagesc' para mapear los valores a colores y 'colormap gray'
figure;
imagesc(imagen_matriz);
colormap(gray);
colorbar;
title('Representación de la Matriz en MATLAB');
axis tight;

[mascara, umbral] = UmbralizaGlobal(imagen_matriz);
new_ima = double(mascara) .* imagen_matriz;

figure;
imshow(new_ima)

figure;
[counts, binLocations]=imhist(new_ima);
bar(binLocations, counts);