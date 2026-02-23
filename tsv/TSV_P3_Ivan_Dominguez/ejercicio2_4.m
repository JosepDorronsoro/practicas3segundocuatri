% Ejercicio 4

clear all;
close all;
clc;

% importamos la imagen

[ima] = imread('P3imagenes/monedas.jpg');
ima = rgb2gray(ima);

[mascara, umbral] = UmbralizaGlobal(ima);

ima_new = uint8(mascara).*ima;

figure; 
subplot(1, 3, 1);
imshow(ima);
title('Imagen original');
subplot(1, 3, 2);
imshow(mascara); % double lo reescala 
title('Máscara');
subplot(1, 3, 3);
imshow(ima_new);
title('Imagen umbralizada')