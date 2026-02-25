% Ejercicio 1

clear all;
close all;
clc;

% Generación de grupos de imágenes

% Primero generamos las matrices correspondientes a x e y:

x = 0:(1/512):1-(1/512);
y = 0:(0.1/256):1-(0.1/256);

[X, Y] = meshgrid(x, y);

% Y ahora grupo por grupo generamos sus funciones y sus 
% figuras correspondientes.

% - - - Primer grupo de funciones - - - %

f1 = 4*X/5;
f2 = Y/2;
f3 = 2*X/5 + Y/4;

figure;
subplot(1, 3, 1);
imshow(f1);
title('$\psi_1(X,Y) = \frac{4X}{5}$', 'Interpreter','latex')
subplot(1, 3, 2);
imshow(f2);
title('$\psi_2(X,Y) = \frac{Y}{5}$', 'Interpreter','latex')
subplot(1, 3, 3);
imshow(f3);
title('$\psi_3(X,Y) = \frac{2X}{5} + \frac{Y}{4}$', 'Interpreter','latex')

% - - - Segundo grupo de funciones - - - %

f1 = cos(2*pi*X);
f2 = cos(4*pi*X);
f3 = cos(8*pi*X);
f4 = cos(16*pi*X);

figure;
subplot(1, 4, 1);
imshow(f1);
title('$\psi_1(X,Y) = cos(2\pi X)$', 'Interpreter','latex')
subplot(1, 4, 2);
imshow(f2);
title('$\psi_2(X,Y) = cos(4\pi X)$', 'Interpreter','latex')
subplot(1, 4, 3);
imshow(f3);
title('$\psi_3(X,Y) = cos(8\pi X)$', 'Interpreter','latex')
subplot(1, 4, 4);
imshow(f4);
title('$\psi_4(X,Y) = cos(16\pi X)$', 'Interpreter','latex')

% - - - Tercer grupo de funciones - - - %

f1 = cos(2*pi*X) + sin(8*pi*Y);
f2 = cos(2*pi*X) + sin(16*pi*Y);

figure;
subplot(1, 2, 1);
imshow(f1);
title('$\psi_1(X,Y) = cos(2 \pi X) + sin(8 \pi Y)$', 'Interpreter','latex')
subplot(1, 2, 2);
imshow(f2);
title('$\psi_2(X,Y) = cos(2 \pi X) + sin(16 \pi Y)$', 'Interpreter','latex')

% - - - Cuarto grupo de funciones - - - %

f1 = cos(4*pi*X + 4*pi*Y);
f2 = cos(4*pi*X + 8*pi*Y);

figure;
subplot(1, 2, 1);
imshow(f1);
title('$\psi_1(X,Y) = cos(4 \pi X + 4 \pi Y)$', 'Interpreter','latex')
subplot(1, 2, 2);
imshow(f2);
title('$\psi_2(X,Y) = cos(4 \pi X + 8 \pi Y)$', 'Interpreter','latex')