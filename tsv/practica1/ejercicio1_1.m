% Ejercicio 1 

clear all;
close all;
clc;

% Con esto podemos cargar las imágenes. 

[ima1, map1] = imread("peppers.png");

[ima2, map2] = imread("cameraman.tif");

[ima3, map3] = imread("corn.tif");

% Para visualizarlas, usaremos el comando
% imshow, cada una lo hará de una forma diferente. 

figure; 
subplot(3, 1, 1);
imshow(ima1)
title('No need to call the map as its RGB true-color'); 
subplot(3, 1, 2);
imshow(ima2)
title('No need to call the map as its true-bw');
subplot(3, 1, 3);
imshow(ima3, map3);
title('As its indexed, we call the colormap');