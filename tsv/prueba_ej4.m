%% Filtrado frecuencial ideal - plantilla de examen
%% Solo tienes que cambiar los parámetros de la sección "PARAMETRIZACIÓN"

clear all; close all; clc;

%% ============================================================
%%  PARAMETRIZACIÓN — solo toca esto
%% ============================================================

% Imagen
ruta_imagen = './P4imas&code/edificio_bw_512.bmp';
M = 512;   % alto de la imagen (filas)
N = 512;   % ancho de la imagen (columnas)

% Frecuencias
fs = 400;       % frecuencia de muestreo
fc = fs / 4;    % frecuencia de corte (prueba también fs/8)

% Tipo de filtro: 1 = cuadrado, 2 = circular, 3 = gaussiano
tipo = 1;

% Tamaño de la máscara pequeña: w => máscara (2w+1)×(2w+1)
% cuadrado/circular: w=1 (3x3), w=2 (5x5), w=3 (7x7)
% gaussiano: w se calcula solo a partir de sigmas (línea ~60)
w = 3;
sigmas = 3;   % solo para gaussiano (número de sigmas de la ventana)

%% ============================================================
%%  FIN PARAMETRIZACIÓN
%% ============================================================

%% Nombre del filtro (para títulos)
switch tipo
    case 1, tipo_str = 'Cuadrado';
    case 2, tipo_str = 'Circular';
    case 3, tipo_str = 'Gaussiano';
end

%% --- BLOQUE 1: construir H del tamaño de la imagen ---

% Retículo frecuencial normalizado [0,1) con M×N puntos
x = (0:N-1) / N;
y = (0:M-1) / M;
[X, Y] = meshgrid(x, y);

% Centrar en (0.5, 0.5) — aquí vive la frecuencia cero
Xc = X - 0.5;
Yc = Y - 0.5;

f_norm = fc / fs;   % frecuencia de corte normalizada

switch tipo
    case 1  % cuadrado
        H = double( (abs(Xc) <= f_norm) & (abs(Yc) <= f_norm) );
    case 2  % circular
        H = double( sqrt(Xc.^2 + Yc.^2) <= f_norm );
    case 3  % gaussiano
        d2 = Xc.^2 + Yc.^2;
        H  = exp( -d2 / (2 * f_norm^2) );
        w  = round( sigmas * (1 / (2*pi*f_norm)) );  % sobreescribe w
end

%% --- BLOQUE 2: obtener la máscara pequeña desde H ---

% Respuesta impulsiva completa (tamaño imagen)
h_desp = real( ifft2( ifftshift(H) ) );
h      = fftshift(h_desp);   % centrar para poder recortar

% Centro de h
cy = floor(M/2) + 1;
cx = floor(N/2) + 1;

% Recorte central (2w+1)×(2w+1)
mascara = h(cy-w : cy+w, cx-w : cx+w);

% Cuantización entera (minimiza MSE, igual que la plantilla original)
[m, ~] = min(mascara(:));
mascara_norm = mascara ./ m;
R = ceil(127 ./ max(abs(mascara_norm(:))));
mse = ones(1, R);
for j = 1:R
    dif = (mascara_norm - round(mascara_norm .* j) ./ j).^2;
    mse(j) = mean(dif(:));
end
[~, j_opt] = min(mse);
mascara_int = round(mascara_norm .* j_opt);
if m < 0
    mascara_int = -mascara_int;
end

% Normalización DC: suma = 1 para no alterar el nivel medio
C = sum(mascara_int(:));
mascara_final = mascara_int ./ C;

%% --- BLOQUE 3: reconstruir H_aprox del tamaño imagen desde la máscara ---

h_aprox = zeros(M, N);
h_aprox(cy-w : cy+w, cx-w : cx+w) = mascara_final;

% Transformada del filtro aproximado (para visualizar)
H_aprox = fftshift( fft2( ifftshift(h_aprox) ) );

%% --- BLOQUE 4: filtrar la imagen con H ideal ---

[ima, ~] = imread(ruta_imagen);
ima = double(ima);

F           = fftshift( fft2(ima) );          % FFT imagen, centrada
G           = F .* H;                         % multiplicación en frecuencia
ima_filtrada = real( ifft2( ifftshift(G) ) ); % vuelta al espacio

%% --- BLOQUE 5: visualización ---

figure('Name', sprintf('Filtrado ideal — %s  [fs=%d  fc=%d  mascara %dx%d]', ...
    tipo_str, fs, fc, 2*w+1, 2*w+1));

% Fila 1 — filtros frecuenciales
subplot(2,3,1);
imagesc(H); colorbar; axis image;
title(sprintf('H ideal (%s)\n[%dx%d]', tipo_str, M, N));

subplot(2,3,2);
imagesc(abs(H_aprox)); colorbar; axis image;
title(sprintf('H aproximado desde máscara\n[%dx%d]', M, N));

subplot(2,3,3);
imagesc(abs(H) - abs(H_aprox)); colorbar; axis image;
title('Diferencia |H| - |H_{aprox}|');

% Fila 2 — máscara pequeña e imagen filtrada
subplot(2,3,4);
imagesc(mascara_final); colorbar; axis image;
title(sprintf('Máscara final (%dx%d)\nnormalizada (suma=1)', 2*w+1, 2*w+1));

subplot(2,3,5);
imshow(uint8(ima)); axis off;
title('Imagen original');

subplot(2,3,6);
imshow(uint8(ima_filtrada)); axis off;
title(sprintf('Imagen filtrada\n(%s, fc=%d)', tipo_str, fc));

% Imprimir máscara en consola para revisarla
fprintf('\n=== Máscara %s (%dx%d), fc=%d, fs=%d ===\n', ...
    tipo_str, 2*w+1, 2*w+1, fc, fs);
disp(mascara_final);
fprintf('Suma coeficientes: %.6f\n', sum(mascara_final(:)));