% P2 Ejercicio 1d

clear all;
close all;
clc;

% 1. Ecuación en diferencias:

% cargamos los coeficientes en un vector:

a=[1 0 0.9025 0 0.8145]; % numerador
b=[1 0 1 0 1]; % denominador

% definimos una delta de longitud 100, por ejemplo:

x=[1, zeros(1, 99)];
h_diff=filter(b, a, x);

% 2. Transformada inversa:

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

h1=conv(hn1, conv(hn2, conv(hn3, hn4))); % numerador 
h2=conv(hd1, conv(hd2, conv(hd3, hd4))); % denominador

% obtenemos la transformada H y su frecuencia en radianes:
[H, w] = freqz(h1, h2, 1024, 'whole'); 
% si no se usa 'whole', perdemos exactitud, pues no se 
% muestrea en frecuencia. 

% y ahora obtenemos su inversa:
h_ifft=real(ifft(H));

% visualizamos
figure;
subplot(2, 1, 1);
stem(1:100, h_diff);
title('h[n] - respuesta al impulso \delta[n]');
xlabel('x[n]'); ylabel('h[n]');
subplot(2, 1, 2);
stem(1:100, h_ifft(1:100));
title('h[n] - inversa de H(Z)');
xlabel('x[n]'); ylabel('h[n]');