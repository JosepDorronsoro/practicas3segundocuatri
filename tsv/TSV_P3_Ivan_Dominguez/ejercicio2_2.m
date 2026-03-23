% Ejercicio 2.2

clear all;
close all;
clc;

% definimos la matriz

matriz = 6 * ones(12, 11);
matriz(4:6, 4:7) = 0;
matriz(10, 8:11) = 0;

% definimos operador promedio

operador_promedio = (1/9) * ones(3,3);

% definimos operador promedio ponderado

aux = ones(3,3);
aux(1,2)=2; aux(2,1)=2; aux(3,2)=2; aux(2,3)=2;
aux(2,2)=4;
operador_promedio_ponderado =(1/16) * aux; 

% obtenemos las imágenes:

ima1 = imfilter(matriz, operador_promedio, 'circular');
ima2 = imfilter(matriz, operador_promedio_ponderado, 'circular');

% definimos el operador de prewitt:

gx = [-1 0 1; -1 0 1; -1 0 1];
gy = [-1 -1 -1; 0 0 0; 1 1 1];

ima3_bordes_x = imfilter(double(matriz), gx, 'circular');
ima3_bordes_y = imfilter(double(matriz), gy, 'circular');

% obtenemos la imagen:

ima3 = sqrt(ima3_bordes_x.^2 + ima3_bordes_y.^2);

figure;

subplot(2, 2, 1);
imshow(matriz, []);
title('Imagen original')

subplot(2, 2, 2);
imshow(ima1, []);
title('Operador promedio')

subplot(2, 2, 3);
imshow(ima2, []);
title('Operador promedio ponderado')

subplot(2, 2, 4);
imshow(ima3, []);
title('Operador de Prewitt')