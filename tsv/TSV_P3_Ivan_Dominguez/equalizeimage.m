function [ima_eq,s] = equalizeimage(ima,nM,nm)

% solo funciona con ima:uint8

[filas, columnas] = size(ima); % para luego
h = imhist(ima); % calculamos histograma
pr = h / (filas*columnas); % n_pixeles / total_pixeles
s = cumsum(pr); % suma los valores (valor entre 0 y 1)
s_escalado = s * (nM-nm) + nm;
ima_eq = uint8(s_escalado(double(ima)+1));

end 