% Ejercicio 3

clear all;
close all;
clc;

% cargamos la imagen:
[ima, map] = imread('P3imagenes/Skin_gray_bc_560.tif');

% la transformamos a imagen de grises:
ima_bw = ind2gray(ima, map); % (un solo canal)

% aplicamos la modificacion de 
% contraste que se pide:

a=80;b=160;s_a=30;s_b=220;
ima_contrast=modificarContraste(ima_bw, a, b, s_a, s_b);

figure;
subplot(2, 2, 1);
imshow(ima);
title('Skin_gray_bc_560.tif');
subplot(2, 2, 2);
imshow(ima_contrast);
title('cambio de contraste de Skin_gray_bc_560.tif')
subplot(2, 2, 3);
[counts, binLocations]=imhist(ima);
bar(binLocations, counts);
subplot(2, 2, 4);
[counts, binLocations]=imhist(ima_contrast);
bar(binLocations, counts);

% Ahora se pide deshacer el cambio. Para ello 
% mantendremos s_a y s_b y disminuiremos el 
% rango [b, a]

% usaremos a=s_a, b=s_b, s_a=a, s_b=b
% para 'reconstruir' la imagen
a=30;b=220;s_a=80;s_b=160; 
ima_contrast_2=modificarContraste(ima_contrast, a, b, s_a, s_b);

figure;
subplot(2, 3, 1);
imshow(ima);
title('Skin_gray_bc_560.tif');
subplot(2, 3, 2);
imshow(ima_contrast);
title('cambio de contraste de Skin_gray_bc_560.tif')
subplot(2, 3, 3);
imshow(ima_contrast_2)
title('rehacemos Skin_gray_bc_560.tif')
subplot(2, 3, 4);
[counts, binLocations]=imhist(ima);
bar(binLocations, counts);
subplot(2, 3, 5);
[counts, binLocations]=imhist(ima_contrast);
bar(binLocations, counts);
subplot(2, 3, 6);
[counts, binLocations]=imhist(ima_contrast_2);
bar(binLocations, counts);