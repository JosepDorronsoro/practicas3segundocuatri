% Ejercicio 5 - tiempos

clear all;
close all;
clc;

[ima] = double(imread('P3imagenes/MRI_gray.jpg'));
[filas_imagen, columnas_imagen] = size(ima);

ventanas_en_vertical = 3; % fijamos este, no podemos cambiarlo

for i=0:7
    
    % calculo de tamaños de bloque

    ventanas_en_horizontal = 2^i;
    M=filas_imagen/ventanas_en_vertical;
    N=columnas_imagen/ventanas_en_horizontal;

    % medicion de tiempos

    tic 

    fun = @(block_struct) (UmbralizaOtsuIntra(block_struct.data)) .* (block_struct.data);
    ima_proc = blockproc(ima, [M N], fun);

    tiempo = toc;
    
    fprintf('Tamaño de bloque (3x%i) - tiempo estimado: %.4f \n', ...
        2^i, tiempo);

end 