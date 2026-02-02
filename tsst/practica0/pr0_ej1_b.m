% añadimos a la ruta los ficheros de audio de la carpeta

addpath(genpath('audios_P0_TSST'));

% cargamos la señal y su frecuencia de muestreo

[adele, fs] = audioread('ADELE_HELLO_53.wav');

% calculamos el numero de muestras y la duracion del audio:

n_samples = length(adele);
duracion = n_samples / fs;

% ahora calculamos el eje tiempo:

eje_t = (0:n_samples-1) / fs;

% y visualizamos la señal de audio 

figure;
limites_ejes=[0 eje_t(end) -0.7 0.7];
plot(eje_t, adele), axis(limites_ejes), grid;
xlabel('Tiempo (s)');
ylabel('Amplitud');
title('Señal de Audio ADELE');

% para seleccionar el tramo de audio 5 segundos,
% seleccionamos el HEEELLOOOO from thee otheeer siiiide

adele_5s = adele(61.5*fs+1:66.5*fs);

% podemos modular cuan rapido se reproduce el segmento 
% aumentando o disminuyendo el factor

factor = 1.7;
sound(adele_5s, (factor*fs));

% por último, podemos seleccionar un tramo de 5 milisegundos 
% para representar la onda con un stem cuyo eje de tiempos sea 
% en milisegundos

% 5ms = 1s / 200

tramo_5ms_adele = adele_5s(2*fs+1:2*fs+fs/200); % 5 ms a partir 
% del segundo 2 del segmento
ejet_5ms = (0:fs/200-1) / fs * 1000; 
figure,stem(ejet_5ms,tramo_5ms_adele)
xlabel('t(ms)')
ylabel('Amplitud')
title('Señal del segmento de 5 milisegundos')