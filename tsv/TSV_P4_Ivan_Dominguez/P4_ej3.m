% Ejercicio 3

clear all;
close all;
clc;

% primero cargamos la imagen como venimos haciendo:

[ima, map]=imread('P4imas&code/edificio_bw_512.bmp');
ima_512=ind2gray(ima,map);

% diseñamos los filtros de media de ordenes 3, 5 y 7:

w3 = (1/3^2)*ones(3,3);
w5 = (1/5^2)*ones(5,5);
w7 = (1/7^2)*ones(7,7);

% filtramos las imagenes:

filtrada3_512=imfilter(ima_512, w3);
filtrada5_512=imfilter(ima_512, w5);
filtrada7_512=imfilter(ima_512, w7);

% las visualizamos:

figure;
subplot(1,3,1), imshow(filtrada3_512), title(sprintf('Filtrada con un filtro de \n media de orden 3'));
subplot(1,3,2), imshow(filtrada5_512), title(sprintf('Filtrada con un filtro de \n media de orden 5'));
subplot(1,3,3), imshow(filtrada7_512), title(sprintf('Filtrada con un filtro de \n media de orden 7'));


[ima, map]=imread('P4imas&code/edificio_bw_1024.bmp');
ima_1024=ind2gray(ima,map);

% las filtramos usando imfilter

filtrada3_1024=imfilter(ima_1024, w3);
filtrada5_1024=imfilter(ima_1024, w5);
filtrada7_1024=imfilter(ima_1024, w7);

% y ahora las filtramos usando imfilter_binomial:

filtrada3_bin_1024=imfilter_binomial(ima_1024, 3);
filtrada5_bin_1024=imfilter_binomial(ima_1024, 5);
filtrada7_bin_1024=imfilter_binomial(ima_1024, 7);

% visualizamos:

figure;
subplot(2,3,1), imshow(filtrada3_1024), 
title(sprintf('Filtrada con un filtro de \n media de orden 3'));
subplot(2,3,2), imshow(filtrada5_1024), 
title(sprintf('Filtrada con un filtro de \n media de orden 5'));
subplot(2,3,3), imshow(filtrada7_1024), 
title(sprintf('Filtrada con un filtro de \n media de orden 7'));
subplot(2,3,4), imshow(filtrada3_bin_1024), 
title(sprintf('Filtrada con un filtro \n binomial de orden 3'));
subplot(2,3,5), imshow(filtrada5_bin_1024), 
title(sprintf('Filtrada con un filtro \n binomial orden 5'));
subplot(2,3,6), imshow(filtrada7_bin_1024), 
title(sprintf('Filtrada con un filtro \n binomial de orden 7'));
