% Ejercicio 1.2 b)

clear all;
close all;
clc;

% Variable independiente:
n = 0:100;

% Señal:

% Primera parte de la señal:
h31 = (0.5).^n;

% Segunda parte de la señal:
h32 = 0.9*(0.5).^(n-1);
h32(1) = 0;

% Resta de ambas:
h3n = h31 - h32;

% Representación:
figure;

subplot(3, 1, 1);
stem(n, h31);
title('h_{31}[n] = (0.5)^n * u[n]')

subplot(3, 1, 2);
stem(n, h32);
title('h_{32}[n] = 0.9(0.5)^{n-1} * u[n-1]')

subplot(3, 1, 3);
stem(n, h3n);
title('h_{3}[n] = (0.5)^n * u[n] - 0.9(0.5)^{n-1} * u[n-1]')