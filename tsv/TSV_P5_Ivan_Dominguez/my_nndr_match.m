function [matches, scores] = my_nndr_match(features1, features2, threshold)
    % my_nndr_match: Asocia características usando el ratio NNDR y una búsqueda greedy
    %
    % Entradas:
    %   features1 - Matriz (N x D) de descriptores de la imagen 1
    %   features2 - Matriz (M x D) de descriptores de la imagen 2
    %   threshold - (Opcional) Umbral de aceptación para NNDR (por defecto 0.8)
    %
    % Salidas:
    %   matches   - Matriz (K x 2) con los índices [idx1, idx2] de las asociaciones
    %   scores    - Vector (K x 1) con los ratios NNDR de cada asociación

    % 1. Configuración inicial
    if nargin < 3
        threshold = 0.8; % Valor recomendado por D. Lowe
    end

    % Convertir a double para garantizar compatibilidad con pdist2
    features1 = double(features1);
    features2 = double(features2);

    % 2. Calcular la matriz de distancias Euclídeas entre todos los descriptores
    % D(i, j) es la distancia de features1(i,:) a features2(j,:)
    % Cálculo manual de distancias euclídeas (reemplazo de pdist2)
    num_f1 = size(features1, 1);
    num_f2 = size(features2, 1);
    D = zeros(num_f1, num_f2);
    for i = 1:num_f1
        % Restamos el descriptor i a todos los de la imagen 2, 
        % elevamos al cuadrado, sumamos por filas y hacemos la raíz
        diffs = features2 - features1(i, :);
        D(i, :) = sqrt(sum(diffs.^2, 2))';
    end

    num_f1 = size(features1, 1);
    potential_matches = zeros(num_f1, 2);
    potential_scores = zeros(num_f1, 1);
    valid_count = 0;

    % 3. Aplicar criterio NNDR
    for i = 1:num_f1
        % Ordenar las distancias del descriptor i a todos los descriptores j
        [sorted_dists, sorted_indices] = sort(D(i, :));

        % Tomar el vecino más cercano (d1) y el segundo más cercano (d2)
        d1 = sorted_dists(1);
        d2 = sorted_dists(2);

        % Evitar división por cero (improbable en SIFT, pero buena práctica)
        if d2 == 0
            ratio = 1; 
        else
            ratio = d1 / d2;
        end

        % Filtrar por umbral NNDR
        if ratio < threshold
            valid_count = valid_count + 1;
            potential_matches(valid_count, :) = [i, sorted_indices(1)];
            potential_scores(valid_count) = ratio;
        end
    end

    % Recortar arrays a las coincidencias reales encontradas
    potential_matches = potential_matches(1:valid_count, :);
    potential_scores = potential_scores(1:valid_count);

    % 4. Búsqueda Greedy para resolver conflictos (asignaciones múltiples)
    % Ordenar los candidatos de mejor a peor ratio (menor es mejor)
    [sorted_scores, sort_idx] = sort(potential_scores);
    sorted_matches = potential_matches(sort_idx, :);

    matches = [];
    scores = [];
    
    % Registro booleano para saber si un punto de la imagen 2 ya fue asignado
    matched_f2 = false(size(features2, 1), 1); 

    for k = 1:length(sorted_scores)
        f1_idx = sorted_matches(k, 1);
        f2_idx = sorted_matches(k, 2);

        % Si el punto en la imagen 2 está libre, hacemos el match
        if ~matched_f2(f2_idx)
            matches = [matches; f1_idx, f2_idx];
            scores = [scores; sorted_scores(k)];
            matched_f2(f2_idx) = true; % Marcarlo como ocupado para futuras iteraciones
        end
    end
end