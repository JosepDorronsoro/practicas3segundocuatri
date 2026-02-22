% Ejercicio 1 - tiempos

clear all;
close all;
clc;

% importamos la imagen

[ima, map] = imread('P3imagenes/edificio_bw_512.bmp');

% Ahora modifiquemos las imágenes según cada aproximación:

L = length(ima);
r = length(map);

% Lo haremos N veces por cada una:

nit = 3000;

% Primera forma: para cada pixel tomamos su 
% (valor máximo) - (valor actual). 

tic
for it=1:nit
ima_proc = r-ima; 
end
tiempo_proc_metodo_1 = toc;

% Segunda forma: implementar la transformación persé.  
r_entrada=0:r-1;
s=r-r_entrada;
ima_proc_2=uint8(s(double(ima)+1));

tic
for it=1:nit
r_entrada=0:r-1;
s=r-r_entrada;
ima_proc_2=uint8(s(double(ima)+1));
end
tiempo_proc_metodo_2 = toc;

% Tercerca forma: consiste en cambiar los valores del map.
% No hay que cambiar el puntero de cada pixel.

% hay que asignar la modificación a cada canal del mapa.
% los valores en map van entre 0 y 1

tic
for it=1:nit
map_proc(:,1)= 1-map(:,1); 
map_proc(:,2)= 1-map(:,2);
map_proc(:,3)= 1-map(:,3);
end
tiempo_proc_metodo_3 = toc;

iname='edificio_bw_512.bmp';
tiempo_p2p = tiempo_proc_metodo_1;
tiempo_valores = tiempo_proc_metodo_2;
tiempo_vlt = tiempo_proc_metodo_3;

fprintf(['Imagen %s (%d iteraciones) - tiempos estimados:\n(1) p2p ' ...
    '    %.4f\n(2) valores %.4f\n(3) vlt %.4f\n'], ...
    iname, nit, tiempo_p2p, tiempo_valores, tiempo_vlt);

% importamos la imagen

[ima, map] = imread('P3imagenes/edificio_bw_1024.bmp');

% Ahora modifiquemos las imágenes según cada aproximación:

L = length(ima);
r = length(map);

% Primera forma: para cada pixel tomamos su 
% (valor máximo) - (valor actual). 

tic
for it=1:nit
ima_proc = r-ima; 
end
tiempo_proc_metodo_1 = toc;

% Segunda forma: implementar la transformación persé.  
r_entrada=0:r-1;
s=r-r_entrada;
ima_proc_2=uint8(s(double(ima)+1));

tic
for it=1:nit
r_entrada=0:r-1;
s=r-r_entrada;
ima_proc_2=uint8(s(double(ima)+1));
end
tiempo_proc_metodo_2 = toc;

% Tercerca forma: consiste en cambiar los valores del map.
% No hay que cambiar el puntero de cada pixel.

% hay que asignar la modificación a cada canal del mapa.
% los valores en map van entre 0 y 1

tic
for it=1:nit
map_proc(:,1)= 1-map(:,1); 
map_proc(:,2)= 1-map(:,2);
map_proc(:,3)= 1-map(:,3);
end
tiempo_proc_metodo_3 = toc;

iname='edificio_bw_1024.bmp';
tiempo_p2p = tiempo_proc_metodo_1;
tiempo_valores = tiempo_proc_metodo_2;
tiempo_vlt = tiempo_proc_metodo_3;

fprintf(['Imagen %s (%d iteraciones) - tiempos estimados:\n(1) p2p ' ...
    '    %.4f\n(2) valores %.4f\n(3) vlt %.4f\n'], ...
    iname, nit, tiempo_p2p, tiempo_valores, tiempo_vlt);