% Ejercicio  1

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

% ahora ploteamos en una ventana:

% - todos
% - 10 elegidos uniformemente 
% - los 10 más fuertes
% - 10 aleatorios


figure;
subplot(2, 2, 1);
imshow(ima); 
hold on; 
h = vl_plotframe_forMatlabSIFTpoints(SIFTpoints); 
set(h, 'Color', 'yellow', 'LineWidth', 2);
plot(SIFTpoints);
hold off;
title('Imagen con todos los puntos SIFT');
subplot(2, 2, 2);
imshow(ima); 
hold on; 
puntos_u = SIFTpoints.selectUniform(10, size(ima));
h = vl_plotframe_forMatlabSIFTpoints(puntos_u); 
set(h, 'Color', 'yellow', 'LineWidth', 2);
plot(puntos_u)
hold off;
title('Imagen con 10 puntos seleccionados uniformemente');
subplot(2, 2, 3);
imshow(ima); 
hold on; 
puntos_s = SIFTpoints.selectStrongest(10);
h = vl_plotframe_forMatlabSIFTpoints(puntos_s); 
set(h, 'Color', 'yellow', 'LineWidth', 2);
plot(puntos_s)
hold off;
title('Imagen con 10 puntos seleccionados según su fuerza');
subplot(2, 2, 4);
imshow(ima); 
hold on; 
totalPuntos = SIFTpoints.Count; 
indices = randperm(totalPuntos, 10); 
puntos_r = SIFTpoints(indices); 
h = vl_plotframe_forMatlabSIFTpoints(puntos_r); 
set(h, 'Color', 'yellow', 'LineWidth', 2);
plot(puntos_r)
title('Imagen con 10 puntos seleccionados aleatoriamente');
hold off;