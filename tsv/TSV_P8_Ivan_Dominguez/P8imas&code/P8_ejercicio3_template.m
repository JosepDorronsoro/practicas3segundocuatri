%% P8 Ej3: detección DT

%% limpiar
clear all;
close all;
clc;

%% inicializar
addpath(genpath('./P8imas&code/'));

ejercicio8_3="Práctica 8 - Ejercicio 3";
%% main
% Cargar dataset INRIA (peatones vs no-peatones)
% Extraer HOG
% Entrenar SVM (fitcsvm)
% Testear (predict)
% 

%% --- Parámetros HOG ---
cellSize = [8 8];
imgSize  = [128 64];   % tamaño típico Dalal-Triggs

% %% --- Rutas dataset ---
dataDir = './P8imas&code/';
posDir = fullfile(dataDir,'inria-person/data_ped/pedestrians');
negDir = fullfile(dataDir,'inria-person/data_ped/no_pedestrians');
posImgs = imageDatastore(posDir);
negImgs = imageDatastore(negDir);

features = [];
labels = [];

quick_train=true;
quick_max_number=400;
% con all falla corredor solo

%% Positivos
if (quick_train)
    number_train_pos=min(quick_max_number,numel(posImgs.Files));
else % all
    number_train_pos=numel(posImgs.Files);
end

for i=1:number_train_pos
    I = readimage(posImgs,i); img=rgb2gray(I); img=im2double(img); img = imresize(img,imgSize);

    feat = extractHOGFeatures(img);

    features = [features; feat];
    labels = [labels; 1];
end

%% Negativos
if (quick_train)
    number_train_neg=min(quick_max_number,numel(negImgs.Files));
else % all
    number_train_neg=numel(negImgs.Files);
end
for i=1:number_train_neg
    I = readimage(negImgs,i); img=rgb2gray(I); img=im2double(img); img = imresize(img,imgSize);

    feat = extractHOGFeatures(img);

    features = [features; feat];
    labels = [labels; -1];
end

%% Entrenar
% para reducir sesgos por orden
rng(42);
idx = randperm(size(features,1));
features = features(idx,:);
labels   = labels(idx);

svmModel = fitcsvm(features, labels, 'KernelFunction','linear', 'Standardize', true);

if (quick_train)
    model_name=sprintf("svmHOG(%d).mat", quick_max_number);
else
    model_name=sprintf("svmHOG(ALL).mat", quick_max_number);
end




%% TEST CROP

% --- Test rápido ---
ITest = imread('Picture_crop.png'); imgTest=rgb2gray(ITest); imgTest=im2double(imgTest); imgTest = imresize(imgTest,imgSize);
featTest = extractHOGFeatures(imgTest);

[label, score] = predict(svmModel, featTest);

%classes = svmModel.ClassNames;
% --- Mostrar imagen ---
figure ('Name', sprintf('%s: test simple', ejercicio8_3));
imshow(imgTest);
title(sprintf('Predicción: %d\n Score(-1):%.4g -  Score(1):%.4g', label, score(1), score(2)));

%% TEST IMAGE (DETECTOR MULTIESCALA y NMS)

files_varias={'Picture.png', 'friends-walking.jpg', 'varias.jpg', 'mix.jpg', 'espaldas.jpg'};
for f=1:size(files_varias,2)
ITest = imread(files_varias{f}); 
imgGray=rgb2gray(ITest); imgGray = im2double(imgGray);
disp(files_varias{f});

cellSize = [8 8];
winSize  = [128 64];

scales = [1 0.9 0.8 0.7 0.6 0.5 0.4];
scales = [1 0.5]; % para completarlo, luego dejar la línea de arriba
bboxes = [];

for s = scales
    
    imgScaled = imresize(imgGray, s);
    [h, w] = size(imgScaled);
    
    % más precision es s peq, menos coste en s grandes
    stride = max(4, round(8*s));
    
    % crear patches de [128 64] en la imagen escalada con paso stride en x e y
    for y = ...
       for x = ...   
            patch = ...;
                        
            feat = extractHOGFeatures(patch);
            
            [label, score] = predict(svmModel, feat);
            
            if score(2) > 0.3 % umbral hard
                % reescalar bbox
                bbox = ...;
                bboxes = [bboxes; bbox score(2)];
            end
        end
    end
end

% seleccionar por score
selected_bboxes=[];
if ~isempty(bboxes)
    boxes = ...; % las bb
    scores = ...; % el score
        
    umbral_score=prctile(scores,95);
        
    figure ('Name', sprintf('%s: test varias', files_varias{f}));
    subplot(1,2,1)
    imshow(ITest); hold on;
    for i=1:size(boxes,1)
        if (scores(i)>umbral_score)
            selected_bboxes = [selected_bboxes; boxes(i,:) scores(i)];
            rectangle('Position',boxes(i,:), 'EdgeColor','r','LineWidth',2);
            x = boxes(i,1); y = boxes(i,2);
            w = boxes(i,3); h = boxes(i,4);
            cx = x + w/2;
            cy = y + h/2;
            text(cx,cy,sprintf('%.2f',scores(i)),'Color','y');
        end
    end
    title(sprintf('Test múltiple: umbral score %.2g)', umbral_score));

end

%% NMS
if ~isempty(selected_bboxes)
    boxes = ...; % las bb
    scores = ...; % el score
        
    umbralIoU=0.3; % más alto IoU → más cajas
    maxBoxes = 4;
    selected = selectStrongestBbox(boxes, scores, 'OverlapThreshold',umbralIoU, 'NumStrongest',maxBoxes);
    
    subplot(1,2,2)
    imshow(ITest); hold on;
    for i=1:size(selected,1)
        rectangle('Position',selected(i,:), 'EdgeColor','r','LineWidth',2);
        x = selected(i,1); y = selected(i,2);
        w = selected(i,3); h = selected(i,4);
        cx = x + w/2;
        cy = y + h/2;
        text(cx,cy,sprintf('%.2f',scores(i)),'Color','y');
    end
    title(sprintf('Test múltiple: umbral score %.2g; NMS umbral IoU %.2g)', umbral_score, umbralIoU));
end

end %for f
