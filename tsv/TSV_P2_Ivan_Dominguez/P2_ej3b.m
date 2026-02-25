% Ejercicio 3b

clear all;
close all;
clc;

% definimos el nuevo espacio con el 
% retículo anterior:

x = 0:(1/200):1-(1/200);
y = 0:(0.1/200):1.03-(0.1/200);

% y definimos la malla:

[X, Y] = meshgrid(x, y);

f1 = sin(40 * pi * X + 30 * pi * Y); 
f2 = sin(120 * X + 90 * Y);

% obtenemos las transformadas:

F1=fftshift(fft2(f1)); % ffshift sirve para que el centro no esté en la esquina
F2=fftshift(fft2(f2));

% creamos los ejes de coordenadas "reales":

image_w = length(x);
image_h = length(y);

fx=-image_w/2:image_w/2; fx=fx(1:end-1);
fy=-image_h/2:image_h/2; fy=fy(1:end-1);

% las pintamos en 2D antes de normalizar:

figure; 
imshow(log(1+abs(F1)),[]);

figure; 
imshow(log(1+abs(F2)),[]);

% Ahora normalizamos los módulos de cada transformada:

F1=abs(F1)/max(max(abs(F1)));
F2=abs(F2)/max(max(abs(F2)));

% y ahora en 3D:

figure;
mesh(fx, fy, F1);

figure;
mesh(fx, fy, F2);