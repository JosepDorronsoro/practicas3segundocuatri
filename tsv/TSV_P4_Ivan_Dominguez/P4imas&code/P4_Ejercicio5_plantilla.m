%% P4 - Ej 5 Diseño frecuencial de máscaras (plantilla)

%% limpiar
clear all;
close all;
clc;

%% inicializar
ejercicio5="Práctica 4 - Ejercicio 5";
addpath(genpath('./P4imas&code'));

%% Parametrización
% tipo de filtro: 1 cuadrado, 2 circular, 3 gaussiano
for tipo=1:3
if (tipo==1)
    tipo_string="Filtro paso bajo ideal cuadrado";
elseif (tipo==2)
    tipo_string="Filtro paso bajo ideal circular";
elseif (tipo==3)
    tipo_string="Filtro paso bajo gaussiano";
end %if

% frecuencia de muestreo
fs=400; % todos

% frecuencia de corte (igual en h y v) 
fc=fs/4; % fs/4; fs/8;

% anchura del filtro
if (tipo==1 || tipo==2) % cuadrado, circular
    w=3; % w=(anchura-1)/2
elseif (tipo==3) % gaussiano
    sigmas=3; % w
    w = round(sigmas*(1/(2*pi*fc/fs)));
end

%% I. Generación filtro frecuencial 

%retículo
v1=...;
v2=...;
x=...;
y=...;
[X,Y]=meshgrid(x,y);

% filtro en posicion central 
f0x=...;
f0y=...;
Xc=...;
Yc=...;

if (tipo==1)
    % filtro paso bajo ideal cuadrado con frecuencia de corte fc
    H =...; 
elseif (tipo==2)
    % filtro paso bajo ideal circular con frecuencia de corte fc
    H =...;
elseif (tipo==3)
    % filtro paso bajo gaussiano con frecuencia de corte fc
    d=sqrt((Xc).^2 + (Yc).^2);
    H=exp(-(d.*d)./(2*fc/fs*fc/fs)); 
end

%% II. Obtención respuesta impulsiva
h_desp_c = ...; 
h_desp = ...;
h = fftshift(h_desp); % por qué es necesario?

%% III. Selección del rango y generación valores enteros

% centro del filtro espacial
cx=...;
cy=...;

% máscara mediante recorte central
filter_mask=h(...);

% Crear version máscara de enteros
% calcular R (máximo factor de ponderación)
[m,j] = min(filter_mask(:));
filter_mask = filter_mask./m; % menor valor = 1, negativos pos, pos negativos
R = ceil(127./max(abs(filter_mask(:))));
% búsqueda de la versión entera que minimiza el MSE
mse = ones(1,R);
for j =1:R
    dif_filtrado = (filter_mask - double(round(filter_mask.*j)./j)).^2;
    mse(j) = mean(dif_filtrado(:));
end
[~,j]=find(mse==min(mse),1);
filter_mask = round(filter_mask.*j);
if (m<0)
    filter_mask=-1.0*filter_mask; % no es si luego se hace el ajuste, pero ...
end

%% IV. Ajuste de la respuesta a una señal constante
C=...; % de teoría
filter_mask=...; 

%% V. Medida error de diseño
% filtro truncado en espacio ("completo" => size(h))
h_t=...;
h_t(...,...)= ...;

% Cálculo filtro truncado en frecuencia
H_T=...; 
%% los shift son equivalentes para pares, pero no para impares (revisar con ejemplo dummy)

% Diferencia en frecuencia (módulo y fase)
DIF_TRUNCADA_MOD=abs(H)-abs(H_T);
DIF_TRUNCADA_PHASE=angle(H)-angle(H_T);

%% VI. Visualizar filtros

% VI.a Visualizar filtros original y truncado

if (tipo==1 || tipo==2 || tipo==4)
    hFig1 = figure('Name', sprintf('%s (H y h_t) - %s [fs=%d fc=%d filtro(%d x %d)]', ejercicio5, tipo_string, fs, fc, 2*w+1,2*w+1) ); 
 elseif (tipo==3)
    hFig1 = figure('Name', sprintf('%s (H y h_t) - %s [fs=%d fc=%d filtro(%d sigmas=>%d x %d)]', ejercicio5, tipo_string, fs, fc, sigmas, 2*w+1,2*w+1) ); 
end

subplot(3,3,1);
imshow(H, [min(H(:)) max(H(:))], 'InitialMagnification', 100); 
title(sprintf('%s H[fx,fy]\nEnergía: %.7g', tipo_string, calcularEnergia(H)))

subplot(3,3,2);
imshow(h, [min(h(:)) max(h(:))], 'InitialMagnification', 100); 
title(sprintf('%s h[x,y]\nEnergía: %.7g', tipo_string, calcularEnergia(h)))

subplot(3,3,3);
plot(x,h(ceil(size(h,1)/2),:)); 
title (sprintf('%s\nPerfil h[n,m], n=ceil(W/2)', tipo_string))

subplot(3,3,4);
imagesc(H);
colorbar;
title(sprintf('%s\nH[fx,fy]\nEnergía: %.7g', tipo_string, calcularEnergia(H)))

subplot(3,3,5);
imagesc(h);
colorbar;
title(sprintf('%s\nh[x,y]\nEnergía: %.7g', tipo_string, calcularEnergia(h)))

subplot(3,3,7)
surf(H,'EdgeColor','None'),
title(sprintf('%s\nH[fx,fy] 3D\nEnergía: %.7g', tipo_string, calcularEnergia(H)))

subplot(3,3,8)
surf(filter_mask,'EdgeColor','None')
title(sprintf('%s\nmáscara(%d x %d) 3D\nEnergía: %.7g', tipo_string, 2*w+1,2*w+1, calcularEnergia(filter_mask)))

% VI.b Visualizar diferencias (error de diseño) en frecuencia

if (tipo==1 || tipo==2 || tipo==4)
    hFig2 = figure('Name', sprintf('%s (Diferencia H y H_T) - %s [fs=%d fc=%d filtro(%d x %d)]', ejercicio5, tipo_string, fs, fc, 2*w+1,2*w+1) ); 
 elseif (tipo==3)
    hFig2 = figure('Name', sprintf('%s (Diferencia H y H_T) - %s [fs=%d fc=%d filtro(%d sigmas=>%d x %d)]', ejercicio5, tipo_string, fs, fc, sigmas, 2*w+1,2*w+1) ); 
end

subplot(2,3,1),
imshow(H, [min(H(:)) max(H(:))]); 
title(sprintf('%s\nH[fx,fy]\nEnergía: %.7g', tipo_string, calcularEnergia(H)))

subplot(2,3,4),
imshow(abs(H_T), [min(abs(H(:))) max(abs(H(:)))]), % if complex only real part shown
title(sprintf('%s\nMod(H[fx,fy]truncada)\nEnergía: %.7g', tipo_string, calcularEnergia(H_T)))

subplot(2,3,2),
imshow(DIF_TRUNCADA_MOD, [min(DIF_TRUNCADA_MOD(:)) max(DIF_TRUNCADA_MOD(:))]); 
title(sprintf('%s\nMod(H[fx,fy]-H[fx,fy]truncada) w=%d\nEnergía: %.7g MSE: %.7g', tipo_string, w, calcularEnergia(DIF_TRUNCADA_MOD), mean(DIF_TRUNCADA_MOD(:))))

subplot(2,3,3),
imagesc(DIF_TRUNCADA_MOD); colorbar %muestra las diferencias en rango ampliado
title(sprintf('%s\nMod(H[fx,fy]-H[fx,fy]truncada) w=%d\nEnergía: %.7g MSE: %.7g', tipo_string, w, calcularEnergia(DIF_TRUNCADA_MOD), mean(DIF_TRUNCADA_MOD(:))))

subplot(2,3,5),
imshow(DIF_TRUNCADA_PHASE, [min(DIF_TRUNCADA_PHASE(:)) max(DIF_TRUNCADA_PHASE(:))]); 
title(sprintf('%s\nFase(H[fx,fy]-H[fx,fy]truncada) w=%d\nEnergía: %.7g MSE: %.7g', tipo_string, w, calcularEnergia(DIF_TRUNCADA_PHASE), mean(DIF_TRUNCADA_PHASE(:))))

subplot(2,3,6),
imagesc(DIF_TRUNCADA_PHASE); colorbar %muestra las diferencias en rango ampliado
title(sprintf('%s\nFase(H[fx,fy]-H[fx,fy]truncada) w=%d\nEnergía: %.7g MSE: %.7g', tipo_string, w, calcularEnergia(DIF_TRUNCADA_PHASE), mean(DIF_TRUNCADA_PHASE(:))))

% VI.c Visualizar efecto filtrado
[ima_original, map]=imread('edificio_bw_512.bmp');

ima_filtrada=((imfilter(double(ima_original),filter_mask)));
dif_filtrado=(double(ima_original)-double(ima_filtrada)).^2;

if (tipo==1 || tipo==2 || tipo==4)
    hFig3 = figure('Name', sprintf('%s (Filtrado) - %s [fs=%d fc=%d filtro(%d x %d)]', ejercicio5, tipo_string, fs, fc, 2*w+1,2*w+1) ); 
 elseif (tipo==3)
    hFig3 = figure('Name', sprintf('%s (Filtrado) - %s [fs=%d fc=%d filtro(%d sigmas=>%d x %d)]', ejercicio5, tipo_string, fs, fc, sigmas, 2*w+1,2*w+1) ); 
end

subplot(1,3,1),
imshow(ima_original),
title(sprintf('ORIGINAL\nEnergía: = %.7g',calcularEnergia(ima_original))), axis off;

subplot(1,3,2),
imshow(uint8(ima_filtrada)),
title(sprintf('IMAGEN FILTRADA\nEnergía = %.7g',calcularEnergia(ima_filtrada))), axis off; 

subplot(1,3,3),
imshow(uint8(dif_filtrado)),
title(sprintf('IMAGEN DIFF^2\nEnergía = %.7g',calcularEnergia(dif_filtrado))), axis off; 

end; % for tipo