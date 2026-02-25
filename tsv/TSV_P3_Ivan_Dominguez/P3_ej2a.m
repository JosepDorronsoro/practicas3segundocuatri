% Ejercicio 2_1

clear all;
close all;
clc;

% importamos la imagen

[ima] = imread('P3imagenes/unequalized.jpg');

figure;
subplot(2, 1, 1);
imshow(ima);
title('unequalized.jpg');
subplot(2, 1, 2);
[counts, binLocations]=imhist(ima);
bar(binLocations, counts);

[ima_stretch, s] = stretchimage(ima, 255, 0);

figure;
subplot(2, 2, 1);
imshow(ima);
title('unequalized.jpg');
subplot(2, 2, 2);
imshow(ima_stretch);
title('stretched unequalized.jpg')
subplot(2, 2, 3);
[counts, binLocations]=imhist(ima);
bar(binLocations, counts);
subplot(2, 2, 4);
[counts, binLocations]=imhist(ima_stretch);
bar(binLocations, counts);