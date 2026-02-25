% Ejercicio 6 

clear all;
close all;
clc;

% Importamos la imagen de peppers:

[ima1_true, map1_true] = imread("peppers.png");

ima1_r = ima1_true(:,:,1);
ima1_g = ima1_true(:,:,2);
ima1_b = ima1_true(:,:,3);

figure; 
subplot(1, 4, 1);
imshow(ima1_true);
title('Imagen original');

subplot(1, 4, 2);
imshow(ima1_r);
title('Imagen r');

subplot(1, 4, 3);
imshow(ima1_g);
title('Imagen g');

subplot(1, 4, 4);
imshow(ima1_b);
title('Imagen b');

% Imagen BRG:

ima1_brg(:,:,1) = ima1_b;
ima1_brg(:,:,2) = ima1_r;
ima1_brg(:,:,3) = ima1_g;

% Imagen RBG:

ima1_rbg(:,:,1) = ima1_r;
ima1_rbg(:,:,2) = ima1_b;
ima1_rbg(:,:,3) = ima1_g;

% Imagen BGR:

ima1_bgr(:,:,1) = ima1_b;
ima1_bgr(:,:,2) = ima1_g;
ima1_bgr(:,:,3) = ima1_r;


figure; 
subplot(1, 3, 1);
imshow(ima1_brg); 
title('imagen brg')
subplot(1, 3, 2);
imshow(ima1_rbg); 
title('imagen rbg')
subplot(1, 3, 3);
imshow(ima1_bgr); 
title('imagen bgr')