% P2 Ejercicio 1a

clear all;
close all;
clc;

% numerador:

hn1=[1 -exp(pi/3*1i)];
hn2=[1 -exp(2*pi/3*1i)];
hn3=[1 -exp(-pi/3*1i)];
hn4=[1 -exp(-2*pi/3*1i)];

% denominador:

hd1=[1 -0.95*exp(pi/3*1i)];
hd2=[1 -0.95*exp(2*pi/3*1i)];
hd3=[1 -0.95*exp(-pi/3*1i)];
hd4=[1 -0.95*exp(-2*pi/3*1i)];

% recordar usar convolución, no el producto de vectores:

h1=conv(hn1, conv(hn2, conv(hn3, hn4)));
h2=conv(hd1, conv(hd2, conv(hd3, hd4)));

% podemos ver el plano Z:

figure;
zplane(h1, h2)

% que debería coincidir con el resultado del polinomio:

hnuevo1=[1 0 1 0 1];
hnuevo2=[1 0 0.9025 0 0.8145];

figure; 
zplane(hnuevo1, hnuevo2)