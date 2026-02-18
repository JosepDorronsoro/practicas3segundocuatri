% ejercicio 6

clear all;
close all;
clc;

% leemos la imagen:

[ima] = imread('cameraman.tif');

FT = fft2(ima);

% en primer lugar, nos quedamos solo con el módulo:

mod = abs(FT);

% y calculamos su inversa:

imagen_reconstruida_1 = real(ifft2(ifftshift(mod)));

% ahora hacemos que el modulo sea cte e igual al valor 
% medio del módulo de la FT

fase = angle(FT);
mod = mean(mean(abs(FT)));
mat_mod = mod * ones(256);

% reconstruimos la FT

FT2 = mod .* exp(1i * fase);

% y calculamos su inversa:

imagen_reconstruida_2 = real(ifft2(ifftshift(FT2)));

figure;
subplot(1, 3, 1); imshow(ima); title('imagen original')
subplot(1, 3, 2); imshow(imagen_reconstruida_1, []); title('imagen sin fase')
subplot(1, 3, 3); imshow(imagen_reconstruida_2, []); title('imagen procesada')