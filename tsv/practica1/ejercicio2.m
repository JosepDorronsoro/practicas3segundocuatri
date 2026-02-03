% Ejercicio 2 

% rango minimo-maximo

figure; 
subplot(2,3,1);
min_ima1=min(ima1(:));
max_ima1= max(ima1(:));
imshow(ima1, [min_ima1, max_ima1]);
colorbar;

subplot(2,3,2);
min_ima2=min(ima2(:));
max_ima2= max(ima2(:));
imshow(ima2, [min_ima2, max_ima2]);
colorbar;

subplot(2,3,3);
min_ima3=min(ima3(:));
max_ima3= max(ima3(:));
imshow(ima3, [min_ima3, max_ima3]);
colorbar;

% rango comprimido

subplot(2,3,4);
imshow(ima1, [min_ima1*2, max_ima1/2]);
colorbar;

subplot(2,3,5);
imshow(ima2, [min_ima2*2, max_ima2/2]);
colorbar;

subplot(2,3,6);
imshow(ima3, [min_ima3*2, max_ima3/3]);
colorbar;