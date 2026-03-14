% Ejercicio 3

clear all;
close all;
clc;

load v_5s
load h_rev
fs=8000;
figure, subplot(211), plot(v_5s), title('Señal de audio'), 
subplot(212), plot(h_rev), title('Respuesta impulsiva de la sala'); 

y_cl=conv(v_5s,h_rev);
figure; 
subplot(211), plot(v_5s), title('Señal original')
subplot(212), plot(y_cl), title('Señal convolucionada con la respuesta impulsiva de la sala'); 

% convolución en frecuencia

N_fft=2^16; 

X=fft(v_5s, N_fft); 
H=fft(h_rev, N_fft);

Y = X(:) .* H(:); 
y_freq = ifft(Y, N_fft);

figure,
subplot(211), plot(y_cl), title('Señal convolucionada con la respuesta impulsiva de la sala'),
subplot(212), plot(y_freq(1:length(y_cl))), title('Señal convolucionada con la respuesta impulsiva de la sala en frecuencia'), axis tight; 

tic
for k=1:3000
y_cl=conv(v_5s,h_rev);
end
t1=toc; 

tic
for k=1:3000
X=fft(v_5s, N_fft); 
H=fft(h_rev, N_fft);
Y = X(:) .* H(:); 
y_freq = ifft(Y, N_fft);
end
t2=toc;