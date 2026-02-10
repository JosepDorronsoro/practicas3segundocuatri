% Ejercicio 4

clear all;
close all;
clc;

% En este ejercicio se pretende grabar 
% las imágenes a disco. Para ello, cargaremos
% las imágenes igual que lo hicimos en el
% anterior ejercicio: 

% - - - - - Imagen 1 - - - - - %

% Formato original (RGB)
[ima1_true, map1_true] = imread("peppers.png");

% Formato indexado (RGB)
[ima1_indexed, map1_indexed] = rgb2ind(ima1_true, 256);

% True BW - Formato RGB

ima1_bw = rgb2gray(ima1_true);

% True BW - Formato indexado

map1_bw = rgb2gray(map1_indexed);

% - - - - - Imagen 2 - - - - - %

% Formato original

[ima2_bw, map2_bw] = imread("cameraman.tif");

% Formato de tres canales RGB

ima2_true (:,:,1)= ima2_bw; 
ima2_true (:,:,2)= ima2_bw; 
ima2_true (:,:,3)= ima2_bw;

% Formato indexado

[ima2_indexed, map2_indexed] = gray2ind(ima2_bw, 256);

% - - - - - Imagen 3 - - - - - %

% Formato indexado

[ima3_indexed, map3_indexed] = imread("corn.tif");

% Formato RGB

ima3_true = ind2rgb(ima3_indexed, map3_indexed); 

% Formato RGB - True BW

ima3_bw = ind2gray(ima3_indexed, map3_indexed); 


% Observando los diferentes tipos, parece claro 
% que las variables de tipo double son:

% - ima3_true

% Nota: la función imwirte espera que los mapas SÍ que 
% sean double, no uint8. 

% Así que las pasamos a uint8 con la funcion im2uint8

ima3_true = im2uint8(ima3_true);

% Y ahora no deberíamos tener problemas grabando las imágenes:

imwrite(ima1_true, 'images/peppers1.tiff') % Correcta 
imwrite(ima1_indexed, map1_indexed, 'images/peppers2.tiff') % Correcta
imwrite(ima1_bw, 'images/peppers3.tiff') % Correcta
imwrite(ima1_indexed, map1_bw, 'images/peppers4.tiff') % Correcta

imwrite(ima2_bw, map2_indexed, 'images/cameraman1.tiff') % Correcta
imwrite(ima2_true, 'images/cameraman2.tiff') % Correcta 
imwrite(ima2_indexed, map2_indexed, 'images/cameraman3.tiff') % Correcta

imwrite(ima3_indexed, map3_indexed, 'images/corn1.tiff') % Correcta
imwrite(ima3_true, 'images/corn2.tiff') % Correcta
imwrite(ima3_bw, 'images/corn3.tiff') % Correcta