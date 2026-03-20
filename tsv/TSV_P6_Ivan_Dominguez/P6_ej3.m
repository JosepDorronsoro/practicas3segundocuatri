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

