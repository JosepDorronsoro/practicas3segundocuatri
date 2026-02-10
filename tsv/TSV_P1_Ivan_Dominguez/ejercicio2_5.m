% Ejercicio 5

clear all;
close all;
clc;

% - - - - - Imagen 1 - - - - - %

[ima1_true, map1_true] = imread("peppers.png");

% Pero la imagen original tendrá que ser de tipo double
% para poder restarle otra imagen. 

ima1_true = im2double(ima1_true);

% Ahora generamos una imagen igual pero de menor granularidad. 
% En concreto, interesa que la granularidad sea muy pequeña
% para que la diferencia sea notable. 

[ima1_indexed, map1_indexed] = rgb2ind(ima1_true, 2);

% Como se ha generado un índice, tendremos que pasarla 
% a rgb de nuevo con la función ind2rgb:

ima1_diff = ind2rgb(ima1_indexed, map1_indexed);

% Y ahora podemos restarlas cómodamente

ima1_error = ima1_true - ima1_diff;

% Y visualizar el resultado de esta operación:

figure; 
subplot(1, 3, 1);
imshow(ima1_true);
title('Imagen original');
subplot(1, 3, 2);
imshow(ima1_diff);
title('Imagen versionada'); 
subplot(1, 3, 3);
imshow(ima1_error);
title('Imagen diferencia. ')

% - - - - - Imagen 2 - - - - - %

[ima2_bw, map2_bw] = imread("cameraman.tif");

% Esta imagen tiene 2 problemas. Solo tiene un canal
% y está en tipo uint8. Primero la pasaremos a double
% para posteriormente crearle sus diferentes canales.

ima2_bw = im2double(ima2_bw);

ima2_true(:,:,1) = ima2_bw; 
ima2_true(:,:,2) = ima2_bw; 
ima2_true(:,:,3) = ima2_bw; 

% Ahora podemos versionarla

[ima2_indexed, map2_indexed] = rgb2ind(ima2_true, 2);

% Y pasar esta versión a rgb:

ima2_diff = ind2rgb(ima2_indexed, map2_indexed);

% Ahora deberíamos poder restarlas:

ima2_error = ima2_true - ima2_diff;

% Y graficarlas:

figure; 
subplot(1, 3, 1);
imshow(ima2_true);
title('Imagen original');
subplot(1, 3, 2);
imshow(ima2_diff);
title('Imagen versionada'); 
subplot(1, 3, 3);
imshow(ima2_error);
title('Imagen diferencia. ')

% - - - - - Imagen 3 - - - - - %

[ima3_indexed, map3_indexed] = imread("corn.tif");

% Esta imagen viene directamente en formato indexado, 
% por lo que tendríamos que pasarla a formato RGB 
% y posteriormente versionarla y operar con ellas. 

ima3_true = ind2rgb(ima3_indexed, map3_indexed);

% Ya viene en formato double. Ahora la versionamos: 

[ima3_indexed_, map3_indexed_] = rgb2ind(ima3_true, 2);
ima3_diff = ind2rgb(ima3_indexed_, map3_indexed_);

% Operamos:

ima3_error = ima3_true - ima3_diff;

% Y las graficamos:

figure; 
subplot(1, 3, 1);
imshow(ima3_true);
title('Imagen original');
subplot(1, 3, 2);
imshow(ima3_diff);
title('Imagen versionada'); 
subplot(1, 3, 3);
imshow(ima3_error);
title('Imagen diferencia. ')