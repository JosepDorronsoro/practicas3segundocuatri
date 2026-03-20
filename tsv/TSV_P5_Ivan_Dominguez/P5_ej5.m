% Ejercicio  5

clear all;
close all;
clc;

% Cargamos la imagen 

addpath(genpath('./P5imas&code/plotSIFT/'));
ima = imread('cameraman.tif');

% Detectamos los puntos con SIFT

SIFTpoints = detectSIFTFeatures(ima); 

% Seleccionamos 10 uniformemente y los 10 más fuertes

puntos_u = SIFTpoints.selectUniform(10, size(ima));
puntos_s = SIFTpoints.selectStrongest(10);

% Los desdoblamos:

[features_u, validPoints_u] = extractFeatures(ima, puntos_u);
[features_s, validPoints_s] = extractFeatures(ima, puntos_s);

% Y graficamos las rejillas de subventanas. 

figure; 
subplot(1, 2, 1);
imshow(ima); hold on;
vl_plotsiftdescriptor_forMatlabSIFTpoints(features_u, points_u); 
title('Points uniform')
subplot(1, 2, 2);
imshow(ima); hold on;
vl_plotsiftdescriptor_forMatlabSIFTpoints(features_s, points_s); 
title('Points strongest')
