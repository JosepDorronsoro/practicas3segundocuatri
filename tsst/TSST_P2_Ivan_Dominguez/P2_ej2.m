% Ejercicio  2

clear all;
close all;
clc;

% La respuesta impulsiva FIR se define por:
naux=0:7;
hfir=[0.95.^(naux), zeros(1, 13)];

% Y la obtenida por transformada inversa:
n=0:20;
h1=0.95.^n .* (n>=0);
h2=0.95^8 .* 0.95 .^ (n - 8) .* (n>8);
h = h1 - h2;

figure
subplot(2, 1, 1); 
stem(n, hfir); title('Respuesta impulsiva FIR');
xlabel('n'); ylabel('h[n]')
subplot(2, 1, 2); 
stem(n, h); title('Respuesta impulsiva mediante TZ inversa');
xlabel('n'); ylabel('h[n]')

% Ahora la representación del filtro (en frecuencia)
% en el plano Z y en módulo y fase

b = [1, zeros(1,7), -0.95^8]; % numerador 
a = [1 -0.95]; % denominador

% plano Z:
figure;
zplane(b,a);

% TZ:
[H, w] = freqz(b, a); 

% Lo representamos usando nuestra función:
modulo_fase(H, w);

% Y el retardo de grupo:

[gd, w]=grpdelay(b,a,1024);

figure;
plot(w/pi, gd);
xlabel('Normalized frequency (\times \pi rad/sample)');
ylabel('Group delay (samples)');
grid on;

% También podría ser como respuesta al impulso de la ecuación en
% diferencias

x=[1, zeros(1, 20)];
h_diff=filter(b, a, x);

figure;
stem(h_diff); title('Respuesta impulsiva por ecuación en diferencias')
xlabel('n'); ylabel('h[n]')

% respuesta del filtro a los tonos:

n=0:5000;

x1 = 1 * cos(pi/20 * n + 0);
x2 = 0.5 * cos(pi/5 * n + pi/3);
x3 = 0.3 * cos(pi/2 * n + pi/5);

x=x1+x2+x3;
y=filter(b,a,x);

figure;
subplot(2, 1, 1);
plot(n(1:200), x(1:200));
xlabel('n'); ylabel('x[n]');
title('señal de entrada');
subplot(2, 1, 2);
plot(n(1:200), y(1:200));
xlabel('n'); ylabel('y[n]');
title('señal de salida')

% y metemos en una figura sus espectros:

[pxx,wx]=pwelch(x);
[pyy,wy]=pwelch(y);

figure;
subplot(2, 1, 1);
plot(wx/pi, 10*log(pxx));
xlabel('Normalized frequency (\times \pi rad/sample)');
ylabel('Potencia/frecuencia (dB/(rad/sample))');
subplot(2, 1, 2);
plot(wy/pi, 10*log(pyy));
xlabel('Normalized frequency (\times \pi rad/sample)');
ylabel('Potencia/frecuencia (dB/(rad/sample))');