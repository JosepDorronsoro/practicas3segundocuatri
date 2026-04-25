%% P8 Ej1: simple/sample Gradient components visualization

%% limpiar
clear all;
close all;
clc;

%% inicializar
addpath(genpath('./P8imas&code/'));

ejercicio8_1='Práctica 8 - Ejercicio 1';
image='Picture_crop.png';
im = imread(image);

% convertir a imagen de gris [0..1]
I = im2gray(im);

figure('Name',sprintf('%s: %s', ejercicio8_1, image)); imshow(I);

%% Gradientes gx - gy

% los kernels nos los dan:

dx = [-1 0 1; -1 0 1; -1 0 1]; 
dy=[-1 -1 -1; 0 0 0; 1 1 1];

% los gradientes son el resulado de convolucionar la
% imagen con el kernel (en cada dirección)

gx=im2double(imfilter(I, dx));
gy=im2double(imfilter(I, dy));

% nota: casting a double para soportar el operador .^
% usamos im2double para reescalar los valores: 
% im2double(ima): [0,...,255] -> [0,1]
% double(ima): [0,...,255] (ints) -> [0,...,255] (doubles)

% Gradiente - magnitud
g = sqrt(gx.^2 + gy.^2);

% Gradiente - orientación

% atan2 -> [-pi, pi]
% atan -> [0, pi]

o = atan2(gy,gx);  
o2 = atan(gy/gx); 

%% Visualizar area 
% seleccionar area
cmin = 26;cmax=cmin+8;
rmin = 10;rmax=rmin+8;

gx_ = gx(rmin:rmax,cmin:cmax);
gy_ = gy(rmin:rmax,cmin:cmax);
x = 1:size(gx_,2);
y = 1:size(gy_,1);
[X, Y] = meshgrid(x, y);
data=I(rmin:rmax,cmin:cmax);

% Visualizar
%winsize= [250 250]; % para verlo ampliado??
figh=figure('Name', sprintf('%s: %s 8x8 block@(%d %d)(%d %d)', ejercicio8_1, image, cmin, rmin, cmax, rmax));


subplot(2,4,1); 
imshow(data);title('Pixels')
print_text_for_each_pixel(figh,data);

subplot(2,4,2); 
datagx=gx(rmin:rmax,cmin:cmax);
imshow(datagx);title('gx')
print_text_for_each_pixel(figh,datagx*255);

subplot(2,4,3); title('gy')
datagy=gy(rmin:rmax,cmin:cmax);
imshow(datagy);title('gy')
print_text_for_each_pixel(figh,datagy*255);

subplot(2,4,4); 
datag=g(rmin:rmax,cmin:cmax);
imshow(datag);title('GMag')
print_text_for_each_pixel(figh,datag*255);

subplot(2,4,5); 
datao=o(rmin:rmax,cmin:cmax);
imshow(datao);title('GOrient 360')
print_text_for_each_pixel(figh,datao*180/pi);

subplot(2,4,6); 
imshow(data); title('Orientación 360'); hold on;
quiver(X,Y,gx_,gy_);

subplot(2,4,7); 
imshow(data); title('Orientación 180'); hold on;
gx_180 = gx_;
gy_180 = gy_;
mask = gy_ < 0;
gx_180(mask) = -gx_(mask);
gy_180(mask) = -gy_(mask);
quiver(X,Y,gx_180,gy_180);

subplot(2,4,8); 
datao2=o2(rmin:rmax,cmin:cmax);
imshow(datao2);title('GOrient 180')
print_text_for_each_pixel(figh,datao2*180/pi);

%% function print_text_for_each_pixel(figureHandle,data)
function print_text_for_each_pixel(figureHandle,data)

set(0, 'CurrentFigure', figureHandle)
hold on;

[m n]=size(data);                    
for i = 1:m
  for j = 1:n
      nu = data(i,j);
      val = num2str(round(nu));
      text(j-0.4,i,val,'color','r')
  end
end
hold off;
end