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

% 2. Transformada inversa con residuez (lo que se pide):

[r, p, k] = residuez(b, a);
n=0:99;
h_res = sum(r .* p.^n) + (n==0)*k; 

% visualizamos
figure;
subplot(2, 1, 1);
stem(1:100, h_diff);
title('h[n] respuesta al impulso \delta[n] usando filter');
xlabel('n'); ylabel('h[n]');
subplot(2, 1, 2);
stem(n, h_res);
title('h[n] construida con residuos');
xlabel('n'); ylabel('h[n]');

% y ahora solo quedaría filtar la señal con filter y conv:

% filter:

% cargamos la señal ruidosa
load ecg_n1.mat
% definimos el eje temporal en que se mueve:
t=0:1/300:(3600/300)-(1/300);
%filtramos la señal:
y = filter(b, a, ecg_n1);

% conv: 

% es trivial ver que y[n]=x[n]*h[n]:

y_conv = conv(ecg_n1, h_res);

figure; 
subplot(2, 1, 1); plot(t, y);
title('Señal filtrada mediante filter.')
xlabel('t(s)'); ylabel('y[n]');
subplot(2, 1, 2); plot(t, y_conv(1:3600));
title('Señal filtrada mediante conv.');
xlabel('t(s)'); ylabel('x[n]*h[n]');