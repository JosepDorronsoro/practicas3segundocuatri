% Ejercicio 1.2 b) RESPUESTA IMPULSIVA DE H(Z)

clear all;
close all;
clc;

% - - - Primera forma: analítica - - - %

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

% - - - Segunda forma: respuesta del sistema al impuslo - - - %

% Ahora se trata de comprobarlo numéricamente:

% Del apartado anterior sabemos que:

b = [1 -0.9];
a = [1 -0.5];

x = [1 zeros(1, 100)];
h_num = filter(b,a,x);

figure;
subplot(2, 1, 1);
stem(n, h3n);
title('Respuesta del sistema analítica.')
subplot(2, 1, 2);
stem(n, h_num);
title('Respuesta del sistema a un impulso. ')

% - - - Tercera forma: residuos - - - %

% Para verificar que H(Z) = -0.8 / (1-0.5z^-1) + 1.8
% hay que saber que H(Z) descompone una función de transferencia. 

[r, p, k] = residuez(b, a);

% Cuya forma seria sum( r / (1 - p z^-1) ) + sum(k). 
% En nuestro caso:

p = 1 -p;
H_res = (r / p) + k;

% Y es fácil ver que efectivamente verifica 
% H(Z) = -0.8 / (1-0.5z^-1) + 1.8

% Y entonces por fuerza deberíamos poder obtener h[n]
% de forma analítica a partir de esta expresión anterior como:

h_res = -0.8*(0.5).^n;
h_res(1) = h_res(1) + 1.8;

figure;
subplot(2, 1, 1);
stem(n, h3n);
title('Transformada Z directa.')
subplot(2, 1, 2);
stem(n, h_res);
title('Transformada Z inversa mediante residuos. ')