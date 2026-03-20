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

[H, w] = freqz(b, a, 512, 'whole');
modulo_fase(H, w);

[gd, w] = grpdelay(b, a);

% Graficamos la respuesta de grupo
figure;
plot(w, gd);
xlabel('Frecuencia (rad/s)');
ylabel('Retardo de grupo (samples)');
title('Retardo de Grupo del Filtro');

% ahora obtenemos la h[n] excitando el sistema con una delta 

n = 0:50; 
delta = [1, zeros(1, 50)]; 
h = filter(b, a, delta); 

% la graficamos

figure; 
stem(n, h); 
title('h[n] excitando el pulso con una delta')

% podemos obtener analíticamente esta expresión:

[r, p, k] = residuez(b, a);

disp('R:');
disp(r);
disp('p:');
disp(p);
disp('k:');
disp(k);

h_inv = ifft(H); 

figure; 
subplot(2, 1, 1);
stem(n, h); 
title('h[n] excitando el pulso con una delta')
subplot(2, 1, 2);
stem(n, h_inv(1:51));
title('h[n] por transformada inversa')

% representamos el tono:

n=0:5000;

x1 = 1 * cos(pi/20 * n + 0);
x2 = 0.5 * cos(pi/5 * n + pi/3);
x3 = 0.3 * cos(pi/2 * n + pi/5);

x=x1+x2+x3;

% primero, conv

y1=conv(x, h);

% segundo, filter
y2=filter(b,a,x);

figure;
subplot(2, 1, 1);
plot(1:251, y1(1:251));
title('salida obtenida mediante ecuación en diferencias')
axis tight
subplot(2, 1, 2);
plot(1:251, y2(1:251));
title('salida obtenida mediante convolucion con h[n]');
axis tight

% espectros en potencia de ambas señales:

figure;
subplot(2, 1, 1);
pwelch(x);
xlabel('Normalized frequency (\times \pi rad/sample)');
ylabel('Potencia/frecuencia (dB/(rad/sample))');
subplot(2, 1, 2);
pwelch(y1);
xlabel('Normalized frequency (\times \pi rad/sample)');
ylabel('Potencia/frecuencia (dB/(rad/sample))');