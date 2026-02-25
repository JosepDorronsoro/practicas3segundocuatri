% Ejercicio 2_2

clear all;
close all;
clc;

% importamos la imagen

[ima] = imread('P3imagenes/unequalized.jpg');

[ima_equalized, s] = equalizeimage(ima, 255, 0);

figure;
subplot(2, 2, 1);
imshow(ima);
title('unequalized.jpg');
subplot(2, 2, 2);
imshow(ima_equalized);
title('equalized unequalized.jpg')
subplot(2, 2, 3);
[counts, binLocations]=imhist(ima);
bar(binLocations, counts);
subplot(2, 2, 4);
[counts, binLocations]=imhist(ima_equalized);
bar(binLocations, counts);