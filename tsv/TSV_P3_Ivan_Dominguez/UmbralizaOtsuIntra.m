function [mascara, umbral] = UmbralizaOtsuIntra(ima)

% ima:uint8. ima:double chilla. 

varianzas = zeros(1,255);
[h,w]=size(ima);

for k=0:255
    umbral_i=k;
    mascara=ima>umbral_i;
    valores_objeto=ima(mascara); valores_fondo=ima(~mascara);

    % para varianza intra-clase:
    p1=sum(sum(mascara))/(h*w); p0=1-p1;
    v0=var(valores_fondo, 1); v1=var(valores_objeto, 1);
    varianzas(k+1)= (p0*v0+p1*v1); 

    % para varianza inter-clase:
    % m0 = mean(valores_fondo); % Media del fondo
    % m1 = mean(valores_objeto); % Media del objeto
    % varianzas(k+1) = p0 * p1 * (m0 - m1)^2;
end

% para varianza intra-clase:
[~, indice]=min(varianzas);

% para varianza intra-clase:
% [~, indice] = max(varianzas;

umbral=indice-1;
mascara=ima>(umbral);

end