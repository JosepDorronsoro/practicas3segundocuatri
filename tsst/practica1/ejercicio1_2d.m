% Generamos una señal discreta de 5001 muestras 
% formada por la suma de los tres tonos:

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

% Ahora calculamos el poder espectral de la señal completa
% en decibelios:

pxx = 20*log10(pwelch(x));

% Y representamos en una figura una gráfica para las 200
% primeras muestras de la señal, y el poder espectral de
% la señal completa:

% HAY QUE HACER ALGUN TIPO DE NORMALIZACIÓN, 
% NO SÉ CUAL

figure;
subplot(2, 1, 1);
plot(1:200, x(1:200));
title('x[n] = x1[n] + x2[n] + x3[n]');
subplot(2, 1, 2);
plot(pxx);
title('Poder espectral de la señal completa.')
xlabel('Frecuencia');
ylabel('Poder espectral (dB)');