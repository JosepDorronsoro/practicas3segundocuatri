% Ejercicio 7

clear all;
close all;
clc;

% leemos la imagen:

[ima] = imread('cameraman.tif');

% obtenemos la DCT de la imagen:

dct_ima = dct2(ima);

% la invertimos: 

inv_dct_ima = idct2(dct_ima);

% calculamos sus energías:

e_ima = calcularEnergia(ima);
e_dct_ima = calcularEnergia(dct_ima); 
e_inv_ima = calcularEnergia(inv_dct_ima);

% visualizamos la imagen original y la imagen reconstruida

figure;
subplot(1, 3, 1), imshow(ima), title('Imagen Original');
subplot(1, 3, 2), imshow(uint8(dct_ima)), title('DCT de la imagen');
subplot(1, 3, 3), imshow(uint8(inv_dct_ima)), title('Imagen Reconstruida');


% ahora trataremos de calcular la energía de cada bloque para ambas
% imágenes:

K = 64;
operacion = @(block_struct) calcularEnergia(block_struct.data);

ima_ = im2double(ima);
bloques_energia_ima = blockproc(ima_, [K K], operacion);

dct_ima_ = im2double(dct_ima);
bloques_energia_dct_ima = blockproc(dct_ima_, [K K], operacion);