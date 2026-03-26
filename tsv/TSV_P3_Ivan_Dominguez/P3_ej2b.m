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

mask_under_50 = ima_equalized < 50; 
mask_between_50_200 = (ima_equalized >= 50) & (ima_equalized <= 200);
mask_above_200 = ima_equalized > 200; 

pseudo_ima(:,:,1) = double(ima_equalized) .* double(mask_under_50);
pseudo_ima(:,:,2) = double(ima_equalized) .* double(mask_between_50_200);
pseudo_ima(:,:,3) = double(ima_equalized) .* double(mask_above_200);

figure;
imshow(pseudo_ima)

