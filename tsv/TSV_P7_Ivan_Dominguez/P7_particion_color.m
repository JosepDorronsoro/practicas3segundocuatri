%% P7 - Ej Segmentación por Partición

%% limpiar
clear all;
close all;
clc;

%% inicializar
addpath(genpath('./P7imas&code/'));

ejercicio1="Práctica 7 - Ejercicio Particiones";

gray = false; % false;

%% --- Cargar imagen ---
[img, map] = imread('tiergarten-schonbrunn.jpg'); % la imagen es rgb => map = []
if not (gray)
    % pasarla a double [0..1]
    img = im2double(img);
end

[M, N, C] = size(img) ; % fils, cols, channels

%% --- Parámetros ---
varThreshold = 0.005*C;   % umbral de varianza para dividir bloques
minSize = 8;              % tamaño mínimo permitido de cada lado del bloque

%% --- Estructura de bloques: [fila, col, alto, ancho] ---
rootBlock = [1, 1, M, N];

blocks = {};     % lista de bloques finales
queue  = {rootBlock};   % cola para procesar (BFS)

%% --- FASE 1: SPLIT ---
while ~isempty(queue)
    blk = queue{1};
    queue(1) = [];   % quitar el primero

    r = blk(1); c = blk(2);
    h = blk(3); w = blk(4);

    if not (gray)
        region_r = img(r:r+h-1, c:c+w-1, 1);
        region_g = img(r:r+h-1, c:c+w-1, 2);
        region_b = img(r:r+h-1, c:c+w-1, 3);
        var_r = var(double(region_r(:)));
        var_g = var(double(region_g(:)));
        var_b = var(double(region_b(:)));
        v = 0.3*var_r + 0.6*var_g + 0.1*var_b; 
   end
    
    % --- Condición para dividir ---
    if v > varThreshold && h > minSize && w > minSize
        % Dividir en 4 rectángulos (rectangulares si w!=h)
        h2 = floor(h/2);
        w2 = floor(w/2);

        % Bloques hijo
        b1 = [r,       c,       h2,       w2];
        b2 = [r,       c+w2,    h2,       w-w2];
        b3 = [r+h2,    c,       h-h2,     w2];
        b4 = [r+h2,    c+w2,    h-h2,     w-w2];

        queue = [queue, {b1, b2, b3, b4}];

    else
        % No dividir → bloque final
        blocks{end+1} = blk;
    end
end

%% --- FASE 2: CALCUALR REPRESENTANTE (variancia) ---
% Convertir a matriz etiquetada para unir regiones similares.
seg = zeros(M, N, C);

for k = 1:length(blocks)

    b = blocks{k};
    r = b(1); c = b(2);
    h = b(3); w = b(4);

    if not (gray)
    region_r = img(r:r+h-1, c:c+w-1, 1);
    region_g = img(r:r+h-1, c:c+w-1, 2);
    region_b = img(r:r+h-1, c:c+w-1, 3);
    seg(r:r+h-1, c:c+w-1, 1) = mean(double(region_r(:)));  
    seg(r:r+h-1, c:c+w-1, 2) = mean(double(region_g(:)));  
    seg(r:r+h-1, c:c+w-1, 3) = mean(double(region_b(:)));  
    end

end % blocks

%% --- Mostrar resultados ---
figure('Name', sprintf('%s\numbral_var %.4g min_tam %d ', ejercicio1, varThreshold, minSize)); 

subplot 231; 
imshow(img, []); colorbar; title(sprintf('Imagen original\nE=%.3g', calcularEnergia(img)));

subplot 232; 
imshow(seg, []); colorbar; title(sprintf('Segmentación Partición Rectangular\nE=%.3g', calcularEnergia(seg)));

subplot 233; 
imshow(img-seg, []); colorbar; title(sprintf('Diferencia\nE=%.3g', calcularEnergia(img-seg)));

subplot 234;
imshow(img, []); colorbar; title('Bloques rectangulares superpuestos');
drawBlockGrid(blocks, 'r', 1);

subplot 235;
imshow(seg, []); colorbar; title('Segmentación + Grid de bloques');
drawBlockGrid(blocks, 'g', 1);

subplot 236;
imshow(img-seg, []); colorbar; title('Diferencia + Grid bloques');
drawBlockGrid(blocks, 'b', 1);
