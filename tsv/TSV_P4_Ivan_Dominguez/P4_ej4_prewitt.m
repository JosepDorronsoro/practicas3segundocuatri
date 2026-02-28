% Ejercicio 4 - Prewitt

clear all;
close all;
clc;

% primero cargamos la imagen como venimos haciendo:

[ima, map]=imread('P4imas&code/edificio_bw_512.bmp');
ima=ind2gray(ima,map);
ima=im2double(ima);

% ahora definimos los filtros de contorno:

s1_aux = ones(3,3);
s1_aux(2,:)=[0 0 0]; s1_aux(3,:)=[-1 -1 -1];
s1 = (1/6) * s1_aux;

s2_aux = ones(3,3);
s2_aux(:,2)=[0 0 0]; s2_aux(:,3)=[-1 -1 -1];
s2 = (1/6) * s1_aux;

% calculamos las imagenes de borde:

ima_borde_1=imfilter(ima, s1);
ima_borde_2=imfilter(ima, s2);

% la intensidad del borde se calcularía como:

F1 = imfilter(ima,s1);
F2 = imfilter(ima,s2);
F  = sqrt(F1.^2 + F2.^2);

% visualizamos con imagesc

figure;
subplot(1, 4, 1); imshow(ima);
title('Imagen Original');
subplot(1, 4, 2); colormap("gray"); imagesc(ima_borde_1);
axis square; title('Bordes horizontales (x)')
subplot(1, 4, 3); colormap("gray"); imagesc(ima_borde_2);
axis square; title('Bordes verticales (y)')
subplot(1, 4, 4); colormap("gray"); imagesc(F);
axis square; title('Gradiente')

