% Ejercicio  3

clear all;
close all;
clc;

addpath('./P6images&code/')

% Cargamos la imagen

[ima, ~] = imread('bricks.jpg');

% La pasamos a escala de grises:

ima_gray = rgb2gray(ima);

load("texture_filters.mat");

figure;
subplot(1, 4, 1);
imshow(texture0);
subplot(1, 4, 2);
imshow(texture135);
subplot(1, 4, 3);
imshow(texture45);
subplot(1, 4, 4);
imshow(texture90);

% terminar

ima_texture0 = imfilter(ima_gray, texture0, "circular");
ima_texture135 = imfilter(ima_gray, texture135, "circular");
ima_texture45 = imfilter(ima_gray, texture45, "circular");
ima_texture90 = imfilter(ima_gray, texture90, "circular");

figure; 
subplot(5, 1, 1);
imshow(ima_gray);
title('Imagen original')
subplot(5, 1, 2);
imshow(ima_texture0);
title('Imagen horizontales')
subplot(5, 1, 3);
imshow(ima_texture135);
title('Imagen diagonal izqda. ')
subplot(5, 1, 4);
imshow(ima_texture45);
title('Imagen diagonal dcha.')
subplot(5, 1, 5);
imshow(ima_texture90);
title('Imagen verticales')

fun = @(block_struct) mean(block_struct.data(:));

ima_texture0_block = blockproc(ima_texture0, [4,9], fun);
ima_texture135_block = blockproc(ima_texture135, [4,9], fun);
ima_texture45_block = blockproc(ima_texture45, [4,9], fun);
ima_texture90_block = blockproc(ima_texture90, [4,9], fun);

figure;
subplot(4, 1, 1);
imshow(ima_texture0_block, [])
subplot(4, 1, 2);
imshow(ima_texture135_block, [])
subplot(4, 1, 3);
imshow(ima_texture45_block, [])
subplot(4, 1, 4);
imshow(ima_texture90_block, [])

% 

umbral = 0.5;

mask_texture0 = ima_texture0 > umbral;
ima_texture0 = double(ima_texture0) .* double(mask_texture0);

ima_texture_135_45 = (ima_texture135 + ima_texture45) / 2;
mask_texture_135_45 = ima_texture_135_45 > umbral;
ima_texture_135_45 = double(ima_texture_135_45) .* mask_texture_135_45;

mask_texture90 = ima_texture90 > umbral;
ima_texture90 = double(ima_texture90) .* double(mask_texture90);

new_ima(:,:,1) = ima_texture0;       % R
new_ima(:,:,2) = ima_texture_135_45; % G
new_ima(:,:,3) = ima_texture90;      % B

figure;
imshow(new_ima)