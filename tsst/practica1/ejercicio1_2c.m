% Ejercicio 1.2 c) Caracterización en frecuencia de H(Z)

clear all;
close all;
clc;

% La función freqz devuelve las respuestas en frecuencia y 
% angulares de un filtro digital concreto. 

b = [1 -0.9];
a = [1 -0.5];

% Es importante no equivocarse con el orden de a y b.

% Ahora podemos hacer uso de freqz para obtener 
% -h: modulo (pasar a dB) y fase (pasar a grados)
% - w: frecuencia (normalizar)

[h, w] = freqz(b, a, 512);

% Hay que tomar el logaritmo de las frecuencias para presentarlas
% en la escala estándar, dB

modulo = 20*log10(abs(h));

% Y la frecuencia suele presentarse en grados, no en radianes:

fase = angle(h)*180/pi;

% Por último normalizamos la frecuencia (dividiendo entre pi):

frecuencia = w / pi;

% El último paso sería obtener el retardo de grupo:

[gd, ] = grpdelay(b, a, 512);

% La frecuencia que devuelve es exactamente la misma que 
% la que ya tenemos. 

% Por último, graficamos:

figure; 
subplot(3, 1, 1);
plot(frecuencia, modulo);
xlabel('Frecuencia normalizada ( / \pi rad)')
ylabel('Módulo (dB).')
subplot(3, 1, 2);
plot(frecuencia, angle(h)*180/pi);
xlabel('Frecuencia normalizada ( / \pi rad)')
ylabel('Fase (grados).')
subplot(3, 1, 3);
plot(frecuencia, gd);
xlabel('Frecuencia normalizada ( / \pi rad)')
ylabel('Retardo de grupo (nº muestras).')