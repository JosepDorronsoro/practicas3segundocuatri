%% P8 Ej2: HOG Matlab

%% limpiar
clear all;
close all;
clc;

%% inicializar
addpath(genpath('./P8imas&code/'));

ejercicio8_2='Práctica 8 - Ejercicio 2';
image='Picture_crop.png';
im = imread(image);

I=rgb2gray(im);
I = im2double(I);
I=imresize(I, [128,64]);

%% calculate HOG Matlab
% default
[featureVectorDEFAULT,hogVisualizationDEFAULT] = extractHOGFeatures(I);
diff_featDEFAULT=featureVectorDEFAULT-featureVectorDEFAULT;
min_diff_featDEFAULT=min(diff_featDEFAULT);
max_diff_featDEFAULT=max(diff_featDEFAULT);

% default rgb
Irgb = im2double(im);
Irgb=imresize(Irgb, [128,64]);
[featureVectorRGB,hogVisualizationRGB] = extractHOGFeatures(Irgb);
diff_featRGB=featureVectorDEFAULT-featureVectorRGB;
min_diff_featRGB=min(diff_featRGB);
max_diff_featRGB=max(diff_featRGB);

% SignedOrientation
[featureVectorSIGNED,hogVisualizationSIGNED] = extractHOGFeatures(I,'UseSignedOrientation',true);
diff_featSIGNED=featureVectorDEFAULT-featureVectorSIGNED;
min_diff_featSIGNED=min(diff_featSIGNED);
max_diff_featSIGNED=max(diff_featSIGNED);

% Dalal-Triggs HOG
[featureVectorFULL, hogVisualizationFULL] = extractHOGFeatures(I, ...
    'CellSize',[8 8], ...
    'BlockSize',[2 2], ...
    'BlockOverlap',[1 1], ...
    'NumBins',9, ...
    'UseSignedOrientation',false);
diff_featFULL=featureVectorDEFAULT-featureVectorFULL;
min_diff_featFULL=min(diff_featFULL);
max_diff_featFULL=max(diff_featFULL);


% Comprativa
figure('Name', sprintf('%s: HOG Matlab Comparativa Features',ejercicio8_2));

subplot(3,4,1)
imshow(I); title('DEFAULT')
hold on;
plot(hogVisualizationDEFAULT);
subplot(3,4,5)
plot(featureVectorDEFAULT); xlim([1 length(featureVectorDEFAULT)]);
subplot(3,4,9)
plot(diff_featDEFAULT); xlim([1 length(diff_featDEFAULT)]);

subplot(3,4,2)
imshow(Irgb); title('RGB')
hold on;
plot(hogVisualizationRGB);
subplot(3,4,6)
plot(featureVectorRGB); xlim([1 length(featureVectorRGB)]);
subplot(3,4,10)
plot(diff_featRGB); xlim([1 length(diff_featRGB)]);

subplot(3,4,3)
imshow(I); title('SIGNED')
hold on;
plot(hogVisualizationSIGNED);
subplot(3,4,7)
plot(featureVectorSIGNED); xlim([1 length(featureVectorSIGNED)]);
subplot(3,4,11)
plot(diff_featSIGNED); xlim([1 length(diff_featSIGNED)]);

subplot(3,4,4)
imshow(I); title('FULL')
hold on;
plot(hogVisualizationFULL);
subplot(3,4,8)
plot(featureVectorFULL); xlim([1 length(featureVectorFULL)]);
subplot(3,4,12)
plot(diff_featFULL); xlim([1 length(diff_featFULL)]);

% Solo visual
figure('Name', sprintf('%s: HOG Matlab Comparativa Visual',ejercicio8_2));
subplot(1,4,1)
imshow(I); title('DEFAULT')
hold on;
plot(hogVisualizationDEFAULT);

subplot(1,4,2)
imshow(Irgb); title('RGB')
hold on;
plot(hogVisualizationRGB);

subplot(1,4,3)
imshow(I); title('SIGNED')
hold on;
plot(hogVisualizationSIGNED);

subplot(1,4,4)
imshow(I); title('FULL')
hold on;
plot(hogVisualizationFULL);