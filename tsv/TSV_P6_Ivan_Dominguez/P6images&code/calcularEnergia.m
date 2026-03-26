function E = calcularEnergia(img)
img=double(img);
E = sum(img(:).^2);
end
