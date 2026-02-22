function [ima_stretch,s] = stretchimage(ima,nM,nm)

% solo funciona con ima:uint8

%calculamos los límites de la imagen
r_min = double(min(ima(:)));
r_max = double(max(ima(:)));
r_entrada=0:255; 
% asumimos que la imagen es uint8 y 
% que tendrá una escala de 256 valores
s = (r_entrada-r_min) .* ( (nM-nm) / (r_max - r_min)) + nm;
% para que los valores no se salgan del rango:
s(s < nm) = nm;
s(s > nM) = nM;
% y visualizamos la imagen:
ima_stretch=uint8(s(double(ima)+1));

end 