%% P7 - Ej Segmentación por kmeans

%% limpiar
clear all;
close all;
clc;

%% inicializar
addpath(genpath('./P7imas&code/'));

ejercicio1="Práctica 7 - Ejercicio kmeans";

gray = false; % cambiar para imagen a color

[img, map] = imread('tiergarten-schonbrunn.jpg'); 
if (gray)
    % pasarla a gris [0..1]
    % rgb2gray -> quita dos canales 
    % im2double no solo cambia el tipado, también la escala
    img = im2double(rgb2gray(img));
else
    img = im2double(img); 
end

% --- Reorganizar la imagen en una matriz de (M*N)xC ---
[m, n, c] = size(img);
pixels = reshape(img, [m*n, c]); 

% --- Elegir el número de clusters ---
K = 4; 

% --- Aplicar kmeans ---
rng(1); % para reproducir resultados iguales para cada imagen/ejecución
[idx, C] = kmeans(pixels, K, 'Distance', 'sqeuclidean', 'Replicates', 4); 
% --- Reconstruir la imagen segmentada ---
segmented_pixels = C(idx,:);
segmented_img = reshape(segmented_pixels, [m, n, c]);

% --- Mostrar resultado ---
figure; 
subplot (3,2,1)
imshow(img); title('Imagen original');

subplot (3,2,3)
imshow(segmented_img); colorbar; title(sprintf('Imagen segmentada con K = %d\nE=%.5g', K, calcularEnergia(segmented_img)));
h324=subplot (3,2,4);
imshow(rgb2gray(segmented_img)); colormap(h324,"hsv"); colorbar; title(sprintf('Imagen segmentada con K = %d\nE=%.5g', K, calcularEnergia(rgb2gray(segmented_img))));

subplot (3,2,5)
diff_double=double(img)-segmented_img;
imshow(diff_double, [-1 1]); colorbar; title(sprintf('Imagen diferencia con K = %d\n[min %.2g, max %.2g]\nE=%.5g', K, -1, 1, calcularEnergia(diff_double)));
subplot (3,2,6)
diff_double=double(img)-segmented_img;
imshow(diff_double, []);  colorbar; title(sprintf('Imagen diferencia con K = %d\n[min %.2g, max %.2g]\nE=%.5g', K, min(diff_double(:)), max(diff_double(:)), calcularEnergia(diff_double)));

C  % displays the centroids values
