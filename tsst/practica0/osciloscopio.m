function osciloscopio(signal,fs,t_ventana,t_refresco)

M = t_ventana * fs;
nv=floor((length(signal)/M));
eje_t_w=(0:M-1)/fs; 
ejes = [eje_t_w(1) eje_t_w(M) 1.05*min(signal) 1.05*max(signal)];

for k=1:nv

    plot(eje_t_w,signal((k-1)*M+1:k*M)),axis(ejes);
    pause(t_refresco);
    
end

