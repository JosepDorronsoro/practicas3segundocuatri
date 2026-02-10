function E = calcularEnergia(ima)
ima=double(ima);
E = sum(ima(:).^2);
end