% Cumpleaños feliz

% definimos los tonos (frecuencias) del cumpleaños feliz, 
% según su partitura real

fs = 44100;

f0_hb=[262 262 294 262 349 330 262 262 294 262 392 349 262 262 523 440 349 330 294 466 466 440 349 392 349];

% y también sus duraciones reales:

d0_hb=[0.5 0.5 1 1 1 2 0.5 0.5 1 1 1 2 0.5 0.5 1 1 1 1 3 0.5 0.5 1 1 1 2];

% mepeamos cada frecuencia con su duración:

hb=[f0_hb;d0_hb];

% y con eso, generaremos la melodía con cada 
% frecuencia y su duración asociadas:

melodia=[];

for k=1:length(hb) % para cada nota de la melodia
    nota = genera_tono(hb(1,k),fs, hb(2,k)); % genera nueva nota
    melodia = [melodia nota]; % añade la nueva nota a la melodia
end

sound(melodia, fs)