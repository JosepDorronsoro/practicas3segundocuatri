% Ejercicio 1

clear all;
close all;
clc;

% Generamos la rejilla: 

x = 0:(1/200):1-(1/200);
y = 0:(0.1/200):1-(0.1/200);

[X, Y] = meshgrid(x, y);

% Ahora representamos las 4 funciones:

f1 = sin(10 * pi * X); 
f2 = sin(20 * pi * Y); 
f3 = sin(40 * pi * X + 30 * pi * Y); 
f4 = sin(10 * pi * X + 20 * pi * Y) + sin(30 * pi * X + 30 * pi * Y); 

figure;
subplot(1, 4, 1);
imshow(f1);
title('$\psi_1(X,Y) = sin(10\pi X)$', 'Interpreter','latex')
subplot(1, 4, 2);
imshow(f2);
title('$\psi_2(X,Y) = sin(20\pi Y)$', 'Interpreter','latex')
subplot(1, 4, 3);
imshow(f3);
title('$\psi_3(X,Y) = sin(40\pi X + 30 \pi Y)$', 'Interpreter','latex')
subplot(1, 4, 4);
imshow(f4);
title('$\psi_4(X,Y) = sin(10\pi X + 20 \pi Y) + sin(30 \pi X + 30 \pi Y)$', 'Interpreter','latex')

% TRANSFORMADAS DE FOURIER

% Representación tridimensional. Primero obtenemos 
% las transformadas:

F1=fftshift(fft2(f1));
F2=fftshift(fft2(f2));
F3=fftshift(fft2(f3));
F4=fftshift(fft2(f4));

% Ahora tendríamos que calcular sus planos de frecuencias. 
% Este plano es común para todas las imágenes, que tienen
% las mismas dimensiones.

image_w = length(x);
image_h = length(y);


fx=-image_w/2:image_w/2; fx=fx(1:end-1);
fy=-image_h/2:image_h/2; fy=fy(1:end-1);

% Ahora normalizamos los módulos de cada transformada:

F1=abs(F1)/max(max(abs(F1)));
F2=abs(F2)/max(max(abs(F2)));
F3=abs(F3)/max(max(abs(F3)));
F4=abs(F4)/max(max(abs(F4)));

figure;
mesh(fx, fy, F1);

figure;
mesh(fx, fy, F2);

figure;
mesh(fx, fy, F3);

figure;
mesh(fx, fy, F4);

% Representación por imagen:

F1=log(1+abs(F1));
figure; 
imshow(F1,[min(min(F1)) max(max(F1))],'InitialMagnification',100);

F2=log(1+abs(F2));
figure; 
imshow(F2,[min(min(F1)) max(max(F2))],'InitialMagnification',100);