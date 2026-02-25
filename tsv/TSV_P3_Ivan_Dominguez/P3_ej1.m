% Ejercicio 1 

clear all;
close all;
clc;

% importamos la imagen

[ima, map] = imread('P3imagenes/edificio_bw_512.bmp');

% La pasamos a true bw:

im_true = ind2rgb(ima , map);

% Y ahora sus histogramas:

[counts, binLocations] = imhist(ima, map);
[counts_, binLocations_] = imhist(im_true);

% Visualizamos los histogramas en una figura 
% junto con sus respectivas imágenes. Las 
% imágenes deberían ser iguales, así como sus 
% histogramas. 

figure;
subplot(2, 2, 1);
imshow(ima, map);
subplot(2, 2, 2);
imshow(im_true);
subplot(2, 2, 3);
bar(binLocations, counts);
title('Histograma de la imagen indexada');
xlabel('Intensidad');
ylabel('Frecuencia');
subplot(2, 2, 4);
bar(binLocations_, counts_)
title('Histograma de la imagen true color');
xlabel('Intensidad');
ylabel('Frecuencia');

% Los histogramas deberían ser exactamente iguales

% Ahora modifiquemos las imágenes según cada aproximación:

L = length(ima);
r = length(map);

% Primera forma: para cada pixel tomamos su 
% (valor máximo) - (valor actual). 

ima_proc_1=r-ima;

% Segunda forma: implementar la transformación persé.  
r_entrada=0:r-1;
s=r-r_entrada;
ima_proc_2=uint8(s(double(ima)+1));

% Tercerca forma: consiste en cambiar los valores del map.
% No hay que cambiar el puntero de cada pixel.

% hay que asignar la modificación a cada canal del mapa.
% los valores en map van entre 0 y 1
map_proc(:,1)= 1-map(:,1); 
map_proc(:,2)= 1-map(:,2);
map_proc(:,3)= 1-map(:,3);

% las visualizamos:

figure;

% primero las imágenes

subplot(2, 4, 1);
imshow(ima);
title('imagen original')
subplot(2, 4, 2);
imshow(ima_proc_1);
title('imagen inversa (método 1)')
subplot(2, 4, 3);
imshow(ima_proc_2);
title('imagen inversa (método 2)')
subplot(2, 4, 4);
imshow(ima, map_proc);
title('imagen inversa (método 3)')

% y luego sus histogramas:

subplot(2, 4, 5);
[counts, binLocations]=imhist(ima, map);
bar(binLocations, counts);
subplot(2, 4, 6);
[counts, binLocations]=imhist(ima_proc_1);
bar(binLocations, counts);
subplot(2, 4, 7);
[counts, binLocations]=imhist(ima_proc_2);
bar(binLocations, counts);
subplot(2, 4, 8);
[counts, binLocations]=imhist(ima, map_proc);
bar(binLocations, counts);