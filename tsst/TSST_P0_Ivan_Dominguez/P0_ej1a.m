% añadimos a la ruta los ficheros de audio de la carpeta

addpath(genpath('audios_P0_TSST'));

% cargamos la señal y su frecuencia de muestreo

[vivaldi, fs] = audioread('vivaldi_invierno.mp3');

% escogemos este audio porque parece ser el menos ruidoso (se
% escucha perfectamente, no como otros)

% nos quedamos solo con la parte mono (es un audio estéreo)

vivaldi_mono = vivaldi(:,1);

% calculamos el numero de muestras y la duracion del audio:

n_samples=length(vivaldi_mono);
duracion=n_samples/fs;

% tambien podemos observar su forma con plot:

figure;
plot(vivaldi_mono);

% pero lo suyo sería mostrar el eje x en unidades de tiempo:

eje_t=(0:n_samples-1)/fs;

% y ahora, volviendo a plotear la señal (con rejilla):

figure;
plot(eje_t, vivaldi_mono); grid

% podemos añadir limites de visualizacion, añadir textos a los ejes 
% y un titulo para que la figura sea algo mas profesional:

figure;
limites_ejes=[0 eje_t(end) -0.5 0.5];
plot(eje_t, vivaldi_mono), axis(limites_ejes), grid
xlabel('t (seg.)'), ylabel('amplitud')
title('Invierno de Vivaldi (segmento)')

% ahora escucharemos un segmento de la cancion 
% (entre los segundos 35 y 45):

% primero seleccionamos el intervalo:

vivaldi_10s = vivaldi_mono(35*fs+1:45*fs);

% podemos representarlo: 

figure;
ejet_10s=(0:length(vivaldi_10s)-1)/fs;
plot(ejet_10s, vivaldi_10s)
xlabel('t (seg.)'),ylabel('amplitud')
title('Invierno de Vivaldi (segmento 10s)'), grid

% escuchamos el segmento seleccionado:

% sound(vivaldi_10s, fs); % descomentar la línea para escuchar

% esto es lo que ejecutamos para escuchar el audio
% con el doble o la mitad de la frecuencia:

factor = 2; % aqui modificamos por 1/2 o por una cte arbitraria

% sound(vivaldi_10s, fs*factor); % descomentar la línea para escuchar

% ahora podemos seleccionar un tramo de 5 milisegundos y visualizarlo:

tramo_5ms_vivaldi = vivaldi_10s(5*fs+1:5*fs+fs/200);
ejet_5ms = (0:fs/200-1) / fs * 1000; 

figure,plot(ejet_5ms,tramo_5ms_vivaldi)
subplot(211),plot(ejet_5ms,tramo_5ms_vivaldi)
subplot(212),stem(ejet_5ms,tramo_5ms_vivaldi)
subplot(211),plot(ejet_5ms,tramo_5ms_vivaldi),grid
xlabel('t (mseg).')
title('Segmento 5 milisegundos de Invierno de Vivaldi')


figure,plot(ejet_5ms,tramo_5ms_vivaldi)