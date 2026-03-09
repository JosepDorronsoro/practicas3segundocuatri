% Generamos una señal discreta de 5001 muestras 
% formada por la suma de los tres tonos:

clear all;
close all;
clc;

% Calculamos cada tono por separado:

n = 0:5000;

x1 = cos(pi / 20 * n);
x2 = 0.5 * cos(pi / 5 * n + pi / 3);
x3 = 0.3 * cos(pi / 2 * n + pi / 5);

% Los sumamos:

x = x1+x2+x3;

% Los graficamos:

figure;
subplot(4, 1, 1);
plot(1:200, x1(1:200));
title('x1[n] = cos(\pi / 20 * n)')
subplot(4, 1, 2);
plot(1:200, x2(1:200));
title('x2[n] = 0.5 * cos(\pi / 5 * n)')
subplot(4, 1, 3);
plot(1:200, x3(1:200));
title('x3[n] = 0.3 * cos(\pi / 2 * n)')
subplot(4, 1, 4);
plot(1:200, x(1:200));
title('x[n] = x1[n] + x2[n] + x3[n]');

% Y representamos en una figura una gráfica para las 200
% primeras muestras de la señal, y el poder espectral de
% la señal completa:

figure;
subplot(2, 1, 1);
plot(1:200, x(1:200));
title('x[n] = x1[n] + x2[n] + x3[n]');
subplot(2, 1, 2);
pwelch(x)

% - - - Señal de salida del filtro correspondiente - - - %

% En primer lugar, obtenemos la respuesta como la convolución
% de la señal respuesta al impulso una delta:

b = [1 -0.5];
a = [1 -0.9];

h = impz(b, a, 200);

y_1 = conv(x, h);
y_1 = y_1(1:length(x));

% Graficamos la señal resultante de la convolución

figure;
plot(1:200, y_1(1:200));
title('Salida de la convolución con el impulso');
xlabel('Muestras');
ylabel('Amplitud');

% Podemos hacer lo mismo con la ecuación en diferencias 
% y sus correspondientes coeficientes de Fourier:

y_2 = filter(b, a, x);

% Graficamos la señal filtrada
figure;
plot(1:200, y_2(1:200));
title('Salida filtrada con los coeficientes de Fourier');
xlabel('Muestras');
ylabel('Amplitud');


% Por último, graficamos en una sola figura: 

% 1. La potencia espectral de la señal
% 2. La respuesta en frecuencia del sistema
% 3. La potencia espectral de la salida

% Primero calculamos la respuesta en frecuencia del sistema
% y la potencia espectral de la salida:

[h, w] = freqz(b, a, 512);
modulo = 20*log10(abs(h));
frecuencia = w / pi;

[Yxx, f_y] = pwelch(y_1);
yxx = 10*log10(Yxx); % para potencia usamos 10*log
f_y = f_y / pi; % normalizamos la frecuencia

% Y ya dibujamos:

figure;
subplot(3, 1, 1);
pwelch(y_1);
subplot(3, 1, 2);
plot(frecuencia, -modulo)
ylabel('dB')
subplot(3, 1, 3);
pwelch(y_2)