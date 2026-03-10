% Ejercicio  3

clear all;
close all;
clc;

% para usar la función vl_plotframe_forMatlabSIFTpoints
% para pintar puntos

addpath(genpath('./P5imas&code/plotSIFT/'));

% importamos la imagen

ima = imread('cameraman.tif');

% detectamos los puntos:

SIFTpoints = detectSIFTFeatures(ima); 
[features, validPoints] = extractFeatures(ima, SIFTpoints);
points = validPoints;

% filtramos los puntos para cada octava:

octava0 = points(points.Scale < 1.6);
octava1 = points(points.Scale >= 1.6 & points.Scale < 3.2);
octava2 = points(points.Scale >= 3.2 & points.Scale < 6.4);
octava3 = points(points.Scale >= 6.4);

figure;

subplot(2, 3, 1);
imshow(ima); 
hold on; 
h = vl_plotframe_forMatlabSIFTpoints(points); 
set(h, 'Color', 'yellow', 'LineWidth', 2);
title(sprintf('Imagen con todos los puntos. %d', length(points)));

subplot(2, 3, 2);
imshow(ima); 
hold on; 
h = vl_plotframe_forMatlabSIFTpoints(octava0); 
set(h, 'Color', 'yellow', 'LineWidth', 2);
title(sprintf('Imagen con los puntos de la primera octava. %d', length(octava0)));

subplot(2, 3, 3);
imshow(ima); 
hold on; 
h = vl_plotframe_forMatlabSIFTpoints(octava1); 
set(h, 'Color', 'yellow', 'LineWidth', 2);
title(sprintf('Imagen con los puntos de la segunda ocatava. %d', length(octava1)));

subplot(2, 3, 4);
imshow(ima); 
hold on; 
h = vl_plotframe_forMatlabSIFTpoints(octava2); 
set(h, 'Color', 'yellow', 'LineWidth', 2);
hold off;
title(sprintf('Imagen con los puntos de la tercera octava. %d', length(octava2)));

subplot(2, 3, 5);
imshow(ima); 
hold on; 
h = vl_plotframe_forMatlabSIFTpoints(octava3); 
set(h, 'Color', 'yellow', 'LineWidth', 2);
hold off;
title(sprintf('Imagen con los puntos de la cuarta octava. %d', length(octava3)));