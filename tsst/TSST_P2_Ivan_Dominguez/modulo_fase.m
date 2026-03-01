function modulo_fase(H, w)

'Grafica, con la escala adecuada, en múdulo y fase, una cierta TZ ';

figure;
subplot(2, 1, 1);
plot(w/pi, 20*log(abs(H)));
xlabel('Normalized frequency (\times \pi rad/sample)');
ylabel('Magnitude')
subplot(2, 1, 2);
plot(w/pi, angle(H)*180/pi);
xlabel('Normalized frequency (\times \pi rad/sample)');
ylabel('Phase (degrees)');

end 