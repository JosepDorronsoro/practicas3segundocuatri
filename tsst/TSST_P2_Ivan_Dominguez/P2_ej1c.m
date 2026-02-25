% P2 Ejercicio 1c

clear all;
close all;
clc;

% cargamos los coeficientes en un vector:
a=[1 0 0.9025 0 0.8145]; %numerador
b=[1 0 1 0 1]; %denominador

% cargamos la señal ruidosa
load ecg_n1.mat
% definimos el eje temporal en que se mueve:
t=0:1/300:(3600/300)-(1/300);
%filtramos la señal:
y = filter(b, a, ecg_n1);

% ploteamos ambas señales para apreciar el 
% efecto del filtro diseñado:

figure;
subplot(2, 1, 1);
plot(t, ecg_n1);
title('ECG con ruido incluido de red de 50Hz'); xlabel('t(s)');
subplot(2, 1, 2);
plot(t, y);
title('ECG tras filtro de Notch'); xlabel('t(s)');

% vemos que se ha eliminado el ruido. 
% podemos volver a comprobar el espectro 
% de potencia para apreciar este efecto 
% con otro formato:

[pxx, wx]=pwelch(ecg_n1);
[pyy, wy]=pwelch(y);
% wx debería ser igual a wy. Visualizamos:
figure;
subplot(2, 1, 1);
plot(wx/pi, 10*log(pxx)); 
title('Espectro de potencia de la señal ruidosa'); 
xlabel('Frecuencia normalizada (\times\pi rad/sample)');
ylabel('Potencia (en dB)');
subplot(2, 1, 2);
plot(wy/pi, 10*log(pyy)); 
title('Espectro de potencia de la señal tras el filtro de Notch'); 
xlabel('Frecuencia normalizada (\times\pi rad/sample)');
ylabel('Potencia (en dB)');