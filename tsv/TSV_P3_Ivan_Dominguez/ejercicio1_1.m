% Ejercicio 1 

[ima, map] = imread('P3imagenes/edificio_bw_512.bmp');

% La pasamos a true color:

im_true = ind2rgb(ima , map);


% Las visualizamos en una figura (deberían ser iguales):

figure;
subplot(1, 2, 1);
imshow(ima, map);
subplot(1, 2, 2);
imshow(im_true);

% Y ahora sus histogramas:

[counts, binLocations] = imhist(ima, map);
[counts_, binLocations_] = imhist(im_true);

% Visualizamos los histogramas en una figura

figure;
subplot(1, 2, 1);
bar(binLocations, counts);
title('Histograma de la imagen indexada');
xlabel('Intensidad');
ylabel('Frecuencia');
subplot(1, 2, 2);
bar(binLocations_, counts_)
title('Histograma de la imagen true color');
xlabel('Intensidad');
ylabel('Frecuencia');

% Los histogramas deberían ser exactamente iguales

% Ahora modifiquemos las imágenes según cada aproximación:

ima_proc=min(ima+c,L-1);

r=[0:L-1];
s=min(r+c,L-1); % definición de la transformación
ima_proc=s(ima+1); % Sumamos 1 porque el valor 0 corresponde a s(1)