function [mascara, umbral] = UmbralizaGlobal(ima)

% Se seguirá el pseudocódigo de las transparencias:

% 1. 𝑇𝑖𝑛𝑖𝑡 = (𝑟𝑚𝑎𝑥 − 𝑟𝑚𝑖𝑛)/2

% 2. Umbralización con 𝑇𝑖𝑛 = 𝑇𝑖𝑛𝑖𝑡, obtención de los niveles medios, 
% 𝑚1 y 𝑚2, de los dos grupos de píxeles que separa el umbral, y 
% generación de un nuevo umbral 𝑇𝑜𝑢𝑡 = (𝑚1 + 𝑚2)/2

% 3. Repetir '2' con 𝑇𝑖𝑛 = 𝑇𝑜𝑢𝑡 hasta que la diferencia entre el 
% nuevo umbral y el anterior sea menor que la unidad.

% 1. 𝑇𝑖𝑛𝑖𝑡 = (𝑟𝑚𝑎𝑥 − 𝑟𝑚𝑖𝑛)/2
r_min = min(ima(:)); r_max=max(ima(:));

% 2. Umbralización con 𝑇𝑖𝑛 = 𝑇𝑖𝑛𝑖𝑡, obtención de los niveles medios, 
% 𝑚1 y 𝑚2, de los dos grupos de píxeles que separa el umbral, y 
% generación de un nuevo umbral 𝑇𝑜𝑢𝑡 = (𝑚1 + 𝑚2)/2
T_init = (r_max-r_min)/2;
T_in = T_init;
m1 = mean(ima(ima<=T_in));
m2 = mean(ima(ima>T_in));
T_out = (m1+m2)/2;

% 3. Repetir '2' con 𝑇𝑖𝑛 = 𝑇𝑜𝑢𝑡 hasta que la diferencia entre el 
% nuevo umbral y el anterior sea menor que la unidad.
while abs(T_in-T_out) > 1
    T_in = T_out;
    m1 = mean(ima(ima<=T_in));
    m2 = mean(ima(ima>T_in));
    T_out = (m1+m2)/2;
end

% Devolvemos el umbral y la máscara.
umbral = T_out;
mascara = ima > umbral;

end