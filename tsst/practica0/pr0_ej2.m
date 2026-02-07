% vamos a generar sintéticamente un tono puro

close all; % Cierra todas las figuras abiertas
fs=44100; % fs=44.1 kHz
f0=500; % Generamos tono de 500 Hz
A=1;

dur=2; % Generamos 2 segundos de tono de 500 Hz
phi=0;
eje_t=(0:fs*dur-1)/fs; % Instantes de tiempo de cada muestra
x_f0 = A * cos(2*pi*f0*eje_t + phi);
sound(x_f0, fs) % Reproducimos el tono de 500 Hz generado
figure;
plot(eje_t, x_f0) % Representamos gráficamente el tono sobre
% un eje de tiempos

N_ciclo=fs/f0; % No tiene por qué ser un número entero
N_5ciclos=ceil(5*N_ciclo); % Tiene que ser número entero


% representamos ahora tan solo los 5 ciclos:

figure;
plot(eje_t(1:N_5ciclos),x_f0(1: N_5ciclos))