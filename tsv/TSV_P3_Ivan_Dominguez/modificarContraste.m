function  [ima_proc,s]=modificarContraste(ima,a,b,s_a,s_b)

% solo funciona con ima:uint8

r = 0:255;
L = length(r);
s = zeros(1, L);

alpha = s_a/a;
beta = (s_b-s_a)/(b-a);
gamma = (L-1-s_b)/(L-1-b);

% en lugar de condicionales, máscaras
% así da igual el valor de r que entre.
mask1 = (r>=0) & (r<a);
mask2 = (r >= a) & (r<b);
mask3 = (r>=b) & (r<L);

% y aplicamos esas máscaras aplicando 
% las fórmulas de tramo:
s(mask1) = alpha * r(mask1);
s(mask2) = beta * (r(mask2)-a) + s_a;
s(mask3) = gamma * (r(mask3)-b) + s_b;
  
% procesamos la imagen:
ima_proc = uint8(s(double(ima)+1));

end