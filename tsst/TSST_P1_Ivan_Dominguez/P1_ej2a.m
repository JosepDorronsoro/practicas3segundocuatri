% Ejercicio 1.2 a)

clear all;
close all;
clc;

% Representamos los polinomios:

h1 = [1 -0.9];
h2 = [1 -0.5];

% Igualamos longitud de los vectores 
% (en este caso no habría hecho falta):

[h1, h2] = eqtflength(h1, h2);
zplane(h1, h2);

% Calculamos numéricamente donde deberían estar los polos:

[z, p, k] = tf2zpk(h1, h2);

% Y con esta función inversa deberíamos obtener los polinomios
% originales, tal que a=h1 y b=h2. 

[a, b] = zp2tf(z, p, k);

