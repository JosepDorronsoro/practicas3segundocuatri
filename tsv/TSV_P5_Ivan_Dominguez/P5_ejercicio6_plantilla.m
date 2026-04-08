%% P6 - Ej 6 Correspondecias SIFT (plantilla)

%% limpiar
clear all;
close all;
clc;

%% inicializar
ejercicio6="Práctica 5 - Ejercicio 6";

addpath(genpath('./P5imas&code/'));

%% leer imágenes (descomentar el "set" que se quiera usar)
file_names={'pupila_1_panorama.jpg', 'pupila_2_panorama.jpg', 'pupila_3_panorama.jpg'};
%file_names={'Mountain1.jpg', 'Mountain2.jpg', 'Mountain2.jpg'}; % en este caso las dos últimas iguales, para probar que se asocian "perfectamente"
%file_names={'I1.jpg', 'I2.jpg', 'I3.jpg'};

for f=1:size(file_names, 2)
% leer ima_gray y asignara a cada vista (1-3)
ima_gray = imread(file_names{f});
ima_gray = im2gray(ima_gray); % convertir a escala de grises
if f==1 ima_gray1=ima_gray;
    elseif f==2 ima_gray2=ima_gray;
    else ima_gray3=ima_gray;
end
end % for f

%% Visualización de imágenes concatenadas
%Concatenar la stres imágenes con "cat"
figure('Name', sprintf('%s Imágenes concatenadas [%s - %s - %s]', ejercicio6, file_names{1}, file_names{2}, file_names{3}))

I_cat=cat(2,ima_gray1,ima_gray1);
I_cat=cat(2,I_cat,ima_gray1);
imshow(uint8(I_cat))
title(sprintf('Imagenes concatenadas de izquierda a derecha  (E = %.4g), 2 (E = %.4g) y 3 (E = %.4g) total (E = %.4g)',calcularEnergia(ima_gray1),calcularEnergia(ima_gray2),calcularEnergia(ima_gray3),calcularEnergia(I_cat)));

%% Detección y extracción caraterísiticas puntos SIFT

SIFTpoints1 = detectSIFTFeatures(ima_gray1);
[features1,valid_SIFTpoints1] = extractFeatures(ima_gray1,SIFTpoints1); 

SIFTpoints2 = detectSIFTFeatures(ima_gray2);
[features2,valid_SIFTpoints2] = extractFeatures(ima_gray2,SIFTpoints2); 

SIFTpoints3 = detectSIFTFeatures(ima_gray3);
[features3,valid_SIFTpoints3] = extractFeatures(ima_gray3,SIFTpoints3); 

%% Opciones Puntos a considerar
% número
points2view=[5, 10, 20, 30, 50];

% modos de selección (1: Strongest; 2: Uniform)
selection_modes=[1,2]; % [5, 10, 20, 30];
selection_mode_names={'strongest', 'uniform'}

for selec=1:size(selection_modes,2)
for p2w=1:size(points2view,2)

% Selección de opciones
npoints2view = points2view(p2w);
if (selec==1)
    % (a1) strongest: Select points with strongest metrics
    selected_points_1=valid_SIFTpoints1.selectStrongest(npoints2view);
    selected_points_2=valid_SIFTpoints2.selectStrongest(npoints2view);
    selected_points_3=valid_SIFTpoints3.selectStrongest(npoints2view);
    elseif (selec==2)
    % (a2) uniform: Select points uniformly
    selected_points_1=valid_SIFTpoints1.selectUniform(npoints2view, size(ima_gray1));
    selected_points_2=valid_SIFTpoints2.selectUniform(npoints2view, size(ima_gray2));
    selected_points_3=valid_SIFTpoints3.selectUniform(npoints2view, size(ima_gray3));
end

%% Trucos "ingenieriles" 
% repetir extract features 
[selected_features_1, selected_points_1]= extractFeatures(ima_gray1,selected_points_1); 
[selected_features_2, selected_points_2]= extractFeatures(ima_gray2,selected_points_2); 
[selected_features_3, selected_points_3]= extractFeatures(ima_gray3,selected_points_3); 
% filtrar al máximo de puntos
selected_points_1=selected_points_1(1:npoints2view);
selected_features_1=selected_features_1(1:npoints2view,:);
selected_points_2=selected_points_2(1:npoints2view);
selected_features_2=selected_features_2(1:npoints2view,:);
selected_points_3=selected_points_3(1:npoints2view);
selected_features_3=selected_features_3(1:npoints2view,:);

%% Visualización de detecciones
figure('Name', sprintf('%s Detectiones SIFT (%d puntos %s) [%s - %s - %s]', ejercicio6, points2view(p2w), selection_mode_names{selec}, file_names{1}, file_names{2}, file_names{3}))

% ima_gray1
subplot(3,2,1); imshow(ima_gray1); hold on;
h12 = plot(selected_points_1, ShowOrientation=true) ; 
title(sprintf('SIFT Matlab (%d puntos %s)\n[%s]',length(selected_points_1), selection_mode_names{selec}, file_names{1}))

subplot(3,2,2); imshow(ima_gray1); hold on
h21=vl_plotsiftdescriptor_forMatlabSIFTpoints(selected_features_1,selected_points_1); set(h21,'color','g'); hold off
title(sprintf('SIFT Matlab descriptores (%d puntos %s)\n[%s]',length(selected_points_1), selection_mode_names{selec}, file_names{1}))

% ima_gray2
subplot(3,2,3); imshow(ima_gray2); hold on;
h12 = plot(selected_points_2, ShowOrientation=true) ;
title(sprintf('SIFT Matlab (%d puntos %s)\n[%s]',length(selected_points_2), selection_mode_names{selec}, file_names{2}))

subplot(3,2,4); imshow(ima_gray2); hold on
h3=vl_plotsiftdescriptor_forMatlabSIFTpoints(selected_features_2,selected_points_2); set(h3,'color','g'); hold off
title(sprintf('SIFT Matlab descriptores (%d puntos %s)\n[%s]',length(selected_points_2), selection_mode_names{selec}, file_names{2}))

% ima_gray3
subplot(3,2,5); imshow(ima_gray3); hold on;
h12 = plot(selected_points_3, ShowOrientation=true) ; 
title(sprintf('SIFT Matlab (%d puntos %s)\n[%s]',length(selected_points_3), selection_mode_names{selec}, file_names{3}))

subplot(3,2,6); imshow(ima_gray3); hold on
h3=vl_plotsiftdescriptor_forMatlabSIFTpoints(selected_features_3,selected_points_3); set(h3,'color','g'); hold off
title(sprintf('SIFT Matlab descriptores (%d puntos %s)\n[%s]',length(selected_points_3), selection_mode_names{selec}, file_names{3}))

%% matching haga uso de la función [indexPairs,matchmetric] = matchFeatures(features1,features2,Name=Value)
% Cálculo correspondencias entre imágenes 1-2
[matches_12, scores_12] = matchFeatures(selected_features_1, selected_features_2);
% Cálculo correspondencias entre imágenes 3-2
[matches_32, scores_32] = matchFeatures(selected_features_3, selected_features_2);

% Seleccione las mejores points2view(p2w) asociaciones 1-2 
% 1. obtenga los índices de los scores ordenados de mayor a menor: función sort
[~,I12]=sort(scores_12, 'ascend');
% 2. seleccione los mejores points3view(p2w) índices
best_i12=I12(1:p2w);
% 3. use los índices mejores para obtener los índices de los matches en 1 y 2
best_matches_12 = matches_12(best_i12, :);
 % 4. seleccione las localizaciones de los puntos 1 y 2 de los matches
selected_pos1_12 = selected_points_1(best_matches_12(:,1));
selected_pos2_12 = selected_points_2(best_matches_12(:,2));

% Seleccione las mejores points2view(p2w) asociaciones 3-2 
% 1. obtenga los índices de los scores ordenados de mayor a menor: función sort
[~,I32]=sort(scores_32, 'ascend');
% 2. seleccione los mejores points3view(p2w) índices
best_i32=I32(1:p2w);
% 3. use los índices mejores para obtener los índices de los matches en 3 y 2
best_matches_32 = matches_32(best_i32, :);
% 4. seleccione las localizaciones de los puntos 1 y 2 de los matches
selected_pos3_32 = selected_points_3(best_matches_32(:,1));
selected_pos2_32 = selected_points_2(best_matches_32(:,2));


%% visualizar matching
figure('Name', sprintf('%s Correspondencias SIFT 1-2 (%d puntos %s) [%s - %s - %s]', ejercicio6, points2view(p2w), selection_mode_names{selec}, file_names{1}, file_names{2}, file_names{3}))

% Visualice la pareja de imágenes 1-2
I12=cat(2,ima_gray1,ima_gray2);
subplot(2,1,1); imshow(uint8(I12)); hold on;
% plot las parejas de puntos 'o'
plot([selected_pos1_12.Location(:,1); selected_pos2_12.Location(:,1)+size(ima_gray1,2)], [selected_pos1_12.Location(:,2); selected_pos2_12.Location(:,2)], 'go', 'MarkerSize', 6);
% plot las líneas entre parejas de puntos '-'
plot([selected_pos1_12.Location(:,1), selected_pos2_12.Location(:,1)+size(ima_gray1,2)]', [selected_pos1_12.Location(:,2), selected_pos2_12.Location(:,2)]', 'y-');
title(sprintf('Mejores %d Correspondencias %d puntos SIFT entre imagen 1 y 2', length(best_i12), points2view(p2w)))

% Visualice la pareja de imágenes 2-3
I23=cat(2,ima_gray2,ima_gray3);
subplot(2,1,2); imshow(uint8(I23)); hold on;
% plot las parejas de puntos 'o'
plot([selected_pos2_32.Location(:,1); selected_pos3_32.Location(:,1)+size(ima_gray2,2)], [selected_pos2_32.Location(:,2); selected_pos3_32.Location(:,2)], 'go', 'MarkerSize', 6);
% plot las líneas entre parejas de puntos '-'
plot([selected_pos2_32.Location(:,1), selected_pos3_32.Location(:,1)+size(ima_gray2,2)]', [selected_pos2_32.Location(:,2), selected_pos3_32.Location(:,2)]', 'y-');
title(sprintf('Mejores %d Correspondencias %d puntos SIFT entre imagen 2 y 3', length(best_i32), points2view(p2w)))

end %for p2w (points2view)
end %for selection_modes

