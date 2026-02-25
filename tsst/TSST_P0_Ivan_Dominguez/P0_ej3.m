% probemos con adele:

addpath(genpath('audios_P0_TSST'));

[signal_adele, fs] = audioread('ADELE_HELLO_53.wav');

tiempo_ventana = 1;

figure;
osciloscopio(signal_adele, fs, tiempo_ventana, 0.05);