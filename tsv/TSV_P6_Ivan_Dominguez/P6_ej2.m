% Ejercicio  2

clear all;
close all;
clc;

addpath('./P6images&code/')

% Cargamos la imagen

[ima, ~] = imread('road_2.jpg');

% La pasamos a escala de grises:

ima_gray = rgb2gray(ima);

% Obtenemos los contornos:

E = edge(ima_gray, 'canny', 0.5, 0.64);

% Visualizamos los bordes con la aproximación:

figure;
imshow(255.*uint8(E)+0.5.*ima_gray);

% La transformada de Hough:

theta = -90:0.5:89.5; % theta conocido
rho_r = 0.5;          % rho fijo, podría ser otro
[H,T,R] = hough(E,'RhoResolution',rho_r,'Theta',theta);

% Visualizamos:

figure; 
imshow(imadjust(mat2gray(H)),'XData',T,'YData',R, 'InitialMagnification','fit');
title('Hough transform');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal; colorbar;


% Los más votados estarán en aquellos índices de T
% que maximicen R, es decir: 

% Lista con el número de máximos que queremos probar
num_maximos_list = [4, 10, 30]; 

for n = num_maximos_list
    % 1. Hacemos una copia de H para poder modificarla (borrar los picos) 
    % sin perder la matriz original para la siguiente prueba de la lista.
    H_aux = H; 
    
    % 2. Preparamos la figura de la imagen original para dibujar las rectas
    fig_img = figure;
    imshow(ima); hold on;
    title(['Rectas detectadas en la imagen (N = ', num2str(n), ')']);
    
    % 3. Preparamos la figura de la transformada para dibujar los máximos
    fig_hough = figure; 
    imshow(imadjust(mat2gray(H)),'XData',T,'YData',R, 'InitialMagnification','fit');
    title(['Máximos en la transformada (N = ', num2str(n), ')']);
    xlabel('\theta'), ylabel('\rho');
    axis on, axis normal; colorbar; hold on;
    
    % 4. Bucle para extraer los 'n' máximos
    for k = 1:n
        % Encontramos el valor máximo global actual
        val_max = max(H_aux(:));
        
        % Encontramos su posición (fila y columna)
        [jrho, jtheta] = find(H_aux == val_max, 1);
        
        % Vamos a la figura de Hough y dibujamos el recuadro verde
        figure(fig_hough);
        plot(T(jtheta), R(jrho), 'sg', 'MarkerSize', 10, 'LineWidth', 2);
        
        % Vamos a la figura de la imagen y dibujamos la recta
        figure(fig_img);
        x = 1:size(E, 2);
        % Calculamos la coordenada 'y' despejando de la ecuación de la recta
        y = (R(jrho) - x.*cosd(T(jtheta))) ./ sind(T(jtheta));
        plot(x, y, '-g', 'LineWidth', 2);
        
        % IMPORTANTE: Anulamos este máximo para que en la próxima iteración 
        % max() encuentre el siguiente pico más alto.
        H_aux(jrho, jtheta) = 0; 
    end
end