% Ejercicio 5

clear all;
close all;
clc;

% Primero definimos la imagen propuesta:

% Para ello, es necesario establecer el retícuclo: 

v1=1/400; v2=1/400;
n=0:v1:1-v1; m=0:v2:1-v2;
[N,M]=meshgrid(n,m);

% Ahora definimos numéricamente la imágen:

f1 = sin(2 * pi * 160 * N) + sin(2 * pi * 40 * M); 

% Obtenemos su su transformada:

F1 = fftshift(fft2(f1));

% definimos el filtro

fcorte_x = 50; fcorte_y = 50; % jugar con la frecuencia de corte

fc_x=fcorte_x*v1; fc_y=fcorte_y*v2;
f0_x=0.5+v1; f0_y=0.5+v2;
fpb=(N>(f0_x-fc_x) & N<(f0_x+fc_x)) & (M>(f0_y-fc_y) & M<(f0_y+fc_y));
fpa=~fpb;
fpb=1*double(fpb); fpa=1*double(fpa);

% Filtramos y hacemos la FT inversa:

F1_pb = F1 .* fpb;
F1_pa = F1 .* fpa;

ima_range=max(max(f1))-min(min(f1)); quant=ima_range/256;
filtered_ima_pa=quant*round(real(ifft2(ifftshift(F1_pa)))/quant);
filtered_ima_pb=quant*round(real(ifft2(ifftshift(F1_pb)))/quant);

figure; 
subplot(1, 3, 1);
imshow(f1, [])
title('Imagen original')
subplot(1, 3, 2);
imshow(filtered_ima_pa, [])
title('Imagen filtrada - paso alto')
subplot(1, 3, 3);
imshow(filtered_ima_pb, [])
title('Imagen filtrada - paso bajo')