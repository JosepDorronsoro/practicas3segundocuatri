% Ejercicio 2

clear all;
close all;
clc;

% cargamos la imagen tal y como en el ejercicio anterior:

[ima, map]=imread('P4imas&code/edificio_bw_512.bmp');
ima=ind2gray(ima,map);


% el filtro de Media-Promediado de orden 3 
% se define como:

w = (1/9)*ones(3,3);

% primero aplicamos el filtro a la imagen:

ima_proc = imfilter(ima, w);

% ahora pasamos ambas a double para poder 
% operar con ellas. no es trivial ver que 
% hay que hacerlo con im2double para reescalar 
% los valores, no se puede hacer con enteros
% ni tampoco usando cast a double (simplemente 
% cambia el tipo de los números sin reescalar)

ima_double = im2double(ima);
ima_proc_double = im2double(ima_proc);

% y ahora calculamos su diferencia al cuadrado

ima_sqrddiff = (ima_proc_double-ima_double).^2;
ima_sqrddiff_int = im2uint8(ima_sqrddiff);

% visualizamos las tres:

figure;
subplot(2, 3, 1);
imshow(ima);
title(sprintf('Imagen Original (uint8) \n energía: %.3e', calcularEnergia(ima))); 
subplot(2, 3, 2);
imshow(ima_proc);
title(sprintf('Imagen procesada, (uint8) \n energía: %.3e', calcularEnergia(ima_proc))); 
subplot(2, 3, 3);
imshow(ima_sqrddiff_int);
title(sprintf('Diferencia al cuadrado, (uint8) \n energía: %.3e', calcularEnergia(ima_sqrddiff_int))); 
subplot(2, 3, 4);
imshow(ima_double);
title(sprintf('Imagen Original (double) \n energía: %.3e', calcularEnergia(ima_double))); 
subplot(2, 3, 5);
imshow(ima_proc_double);
title(sprintf('Imagen procesada, (double) \n energía: %.3e', calcularEnergia(ima_proc_double))); 
subplot(2, 3, 6);
imshow(ima_sqrddiff);
title(sprintf('Diferencia al cuadrado, (double) \n energía: %.3e', calcularEnergia(ima_sqrddiff))); 

