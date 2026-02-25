% Ejercicio 4

clear all;
close all;
clc;

% importamos la imagen

[ima] = imread('P3imagenes/monedas.jpg');
ima = rgb2gray(ima);

% Umbralización global

% obtenemos la máscara y el umbral

[mascara_global, umbral] = UmbralizaGlobal(ima);

% obtenemos la imagen umbralizada

ima_global = uint8(mascara_global).*ima;

% visualizamos:

figure; 
subplot(2, 3, 1);
imshow(ima);
title(sprintf('Imagen original, energía: %.3e', calcularEnergia(ima)));
subplot(2, 3, 2);
imshow(mascara_global);
title(sprintf('Máscara, energía: %.3e', calcularEnergia(mascara_global)));
subplot(2, 3, 3);
imshow(ima_global);
title(sprintf('Imagen umbralizada, energía: %.3e', calcularEnergia(ima_global)));
subplot(2, 3, 4);
[counts, binLocations]=imhist(ima); 
bar(binLocations, counts); axis tight;
subplot(2, 3, 5);
[counts, binLocations]=imhist(mascara_global);
bar(binLocations, counts);
subplot(2, 3, 6);
[counts, binLocations]=imhist(ima_global);
bar(binLocations, counts);

% Umbralización Otsu-Intra

% obtenemos la máscara y el umbral

[mascara_otsu, umbral] = UmbralizaGlobal(ima);

% obtenemos la imagen umbralizada

ima_ostu = uint8(mascara_otsu).*ima;

% visualizamos:

figure; 
subplot(2, 3, 1);
imshow(ima);
title(sprintf('Imagen original, energía: %.3e', calcularEnergia(ima)));
subplot(2, 3, 2);
imshow(mascara_otsu);
title(sprintf('Máscara, energía: %.3e', calcularEnergia(mascara_otsu)));
subplot(2, 3, 3);
imshow(ima_ostu);
title(sprintf('Imagen umbralizada, energía: %.3e', calcularEnergia(ima_ostu)));
subplot(2, 3, 4);
[counts, binLocations]=imhist(ima); 
bar(binLocations, counts); axis tight;
subplot(2, 3, 5);
[counts, binLocations]=imhist(mascara_otsu);
bar(binLocations, counts);
subplot(2, 3, 6);
[counts, binLocations]=imhist(ima_ostu);
bar(binLocations, counts);