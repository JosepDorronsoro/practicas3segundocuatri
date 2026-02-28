% Ejercicio 1

clear all;
close all;
clc;

% cargamos la imagen:

[ima, map]=imread('P4imas&code/edificio_bw_512.bmp');

% es importante pasarla a escala de grises 

ima=ind2gray(ima,map);

% definimos el filtro:

m=zeros(7, 7);
vec=[7 6 4 1 4 6 7];
m(4,:)=vec; C=sum(vec);
w=(1/C)*m;

% procesamos la imagen:

ima_proc=imfilter(ima, w);

% la comparamos con la original:

figure;
subplot(1, 2, 1);
imshow(ima);
title('Imagen sin procesar');
subplot(1, 2, 2);
imshow(ima_proc);
title('Imagen procesada');


