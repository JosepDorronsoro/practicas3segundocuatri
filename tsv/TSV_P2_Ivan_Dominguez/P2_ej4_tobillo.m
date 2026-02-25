% Ejercicio 4 - tobillo

clear all;
close all;
clc;

% cargamos la imagen del tobillo: 

[ima, map] = imread("P2imagenes/RNM-TOBILLO.jpg");

% la pasamos a double:

ima = double(ima);

% calculamos su extension especular:

fila_superior = [ima, fliplr(ima)];
fila_inferior = flipud(fila_superior);
ima_especular = [fila_superior; fila_inferior];

% calculamos sus ft:

FT_normal = fftshift(fft2(ima));
FT_extendida = fftshift(fft2(ima_especular));

% normalizamos las ft:

%FT_normal=abs(FT_normal)/max(max(abs(FT_normal)));
%FT_extendida=abs(FT_extendida)/max(max(abs(FT_extendida)));


% vemos la ft de ambas imágenes:

figure;
subplot(2, 2, 1);
imshow(ima, []);
subplot(2, 2, 2);
imshow(ima_especular, []);
subplot(2, 2, 3);
imshow(log(1+abs(FT_normal)), []);
subplot(2, 2, 4);
imshow(log(1+abs(FT_extendida)), []);
