% Ejercicio 5 

clear all;
close all;
clc;

% Siguiendo el procedimiento del enunciado:

% primero importamos nuestra imagen:

% es importante castear a double de esta forma 
% para que no se escalen los valores. simplemente
% cambian de tipo 128->128.0

[ima] = double(imread('P3imagenes/MRI_gray.jpg'));

% ahora obtenemos el número de filas y columnas:

[filas_imagen, columnas_imagen] = size(ima);

% y ahora un número arbitrario de ventanas en 
% vertical y horizontal:

ventanas_en_vertical = 3; 
ventanas_en_horizontal = 8;

M=filas_imagen/ventanas_en_vertical;
N=columnas_imagen/ventanas_en_horizontal;

% ahora con la funcion blockproc aplicaremos umbralización
% Otsu-Intra en cada uno de los bloques:

% por alguna razon block_struct no acepta multiplicaciones 
% de enteros con doubles. hay que pasarlo a logical

                      % máscara                                  % ima
fun = @(block_struct) (UmbralizaOtsuIntra(block_struct.data)) .* (block_struct.data);
ima_proc = blockproc(ima, [M N], fun);

% personalmente, quiero ver la diferencia entre umbralizar por bloques
% y sin ellos. para ello umbralizamos normal:

[mascara, umbral] = UmbralizaOtsuIntra(ima);
ima_otsu = logical(mascara) .* ima;

ima = uint8(ima);

figure;
subplot(1, 3, 1); imshow(ima);
title('Imagen original')
subplot(1, 3, 2); imshow(ima_otsu);
title('Imagen umbralizada (normal)');
subplot(1, 3, 3); imshow(ima_proc);
title('Imagen umbralizada (por bloques)');