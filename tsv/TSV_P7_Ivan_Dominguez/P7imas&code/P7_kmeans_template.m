%% P7 - Ej Segmentación por kmeans

%% limpiar
clear all;
close all;
clc;

%% inicializar
addpath(genpath('./P7imas&code/'));

ejercicio1="Práctica 7 - Ejercicio kmeans";

gray = true; % false;

% --- Cargar imagen ---
[img, map] = imread('tiergarten-schonbrunn.jpg'); % la imagen es rgb => map = []
if (gray)
    % pasarla a gris [0..1]
    img = rgb...;
end

% --- Reorganizar la imagen en una matriz de (M*N)xC ---
[m, n, c] = ...;
pixels = reshape(img, ...); 

% --- Elegir el número de clusters ---
K = 4; 

% --- Aplicar kmeans ---
rng(1); % para reproducir resultados iguales para cada imagen/ejecución
[idx, C] = kmeans(pixels, K, 'Distance', 'sqeuclidean', 'Replicates', 4); 
% --- Reconstruir la imagen segmentada ---
segmented_pixels = C(...);
segmented_img = reshape(segmented_pixels, ...);


% --- Mostrar resultado ---
figure; 
subplot (3,2,1)
imshow(img); title('Imagen original');

subplot (3,2,3)
imshow(segmented_img); colorbar; title(sprintf('Imagen segmentada con K = %d\nE=%.5g', K, calcularEnergia(segmented_img)));
h324=subplot (3,2,4);
imshow(segmented_img); colormap(h324,"hsv"); colorbar; title(sprintf('Imagen segmentada con K = %d\nE=%.5g', K, calcularEnergia(segmented_img)));

subplot (3,2,5)
diff_double=double(img)-segmented_img;
imshow(diff_double, [-1 1]); colorbar; title(sprintf('Imagen diferencia con K = %d\n[min %.2g, max %.2g]\nE=%.5g', K, -1, 1, calcularEnergia(diff_double)));
subplot (3,2,6)
diff_double=double(img)-segmented_img;
imshow(diff_double, []);  colorbar; title(sprintf('Imagen diferencia con K = %d\n[min %.2g, max %.2g]\nE=%.5g', K, min(diff_double(:)), max(diff_double(:)), calcularEnergia(diff_double)));

C  % displays the centroids values

