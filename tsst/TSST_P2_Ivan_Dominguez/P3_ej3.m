% Ejercicio 3 

clear all;
close all;
clc;

% definimos el filtro

b=conv(0.05634, [1, -0.0166, -0.0166, 1]); % numerador 
a=[1, -2.1291, 1.7834, -0.5435]; % denominador 

figure;
zplane(b,a)

[z,p,k]=tf2zpk(b,a);

% Los pasamos por consola:

disp('Ceros:');
disp(z);
disp('Polos:');
disp(p);
disp('Ganancia:');
disp(k);

% modulo y fase:

[H, w] = freqz(b, a);
modulo_fase(H, w);

