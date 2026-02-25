function x_f0=genera_tono(f0,fs,dur)

% amplitud menor que 1 para evitar saturaciones
A=0.7;
% fase inicial 0 pero con cuidado con otros 
% tonos con fase distinta (podrían cancelarse los tonos)
phi=0; 
% instantes de tiempo de cada muestra
eje_t=(0:fs*dur-1)/fs; 
% devolvemos la señal
x_f0 = A * cos(2*pi*f0*eje_t + phi);
