% Ejercicio  5

clear all;
close all;
clc;

addpath(genpath('./P5imas&code/plotSIFT/'));
ima = imread('cameraman.tif');

SIFTpoints = detectSIFTFeatures(ima); 

puntos_u = SIFTpoints.selectUniform(10, size(ima));
puntos_s = SIFTpoints.selectStrongest(10);

[features_u, validPoints_u] = extractFeatures(ima, puntos_u);
[features_s, validPoints_s] = extractFeatures(ima, puntos_s);

points_u = validPoints_u;
points_s = validPoints_s;

figure; 
subplot(1, 2, 1);
imshow(ima); hold on;
h = vl_plotframe_forMatlabSIFTpoints(points_u); 
set(h, 'Color', 'yellow', 'LineWidth', 2);
title('Points uniform')
subplot(1, 2, 2);
imshow(ima); hold on;
h = vl_plotframe_forMatlabSIFTpoints(points_s); 
set(h, 'Color', 'yellow', 'LineWidth', 2);
title('Points strongest')
