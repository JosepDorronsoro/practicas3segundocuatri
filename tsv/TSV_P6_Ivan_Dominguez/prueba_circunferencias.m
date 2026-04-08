% 1. Cargar imagen y detectar bordes
addpath('./P6images&code')

I = imread('monedas.jpg'); 
I_gray = rgb2gray(I);
bordes = edge(I_gray, 'canny'); % Imagen binaria con los bordes

% 2. Parámetros conocidos
R = 25; % Asumimos que buscamos círculos de radio 30 píxeles
[filas, columnas] = size(bordes);

% 3. Inicializar la matriz de acumulación
acumulador = zeros(filas, columnas);

% Precalcular senos y cosenos para mayor velocidad
theta = 0 : 0.1 : 2*pi; % Resolución del ángulo
cos_theta = cos(theta);
sin_theta = sin(theta);

% 4. Proceso de Votación (El núcleo del método de Hough)
for y = 1:filas
    for x = 1:columnas
        
        % Si el píxel es un borde, generamos sus votos
        if bordes(y, x) == 1 
            for k = 1:length(theta)
                % Calcular coordenadas en el espacio de parámetros (a, b)
                a = round(x - R * cos_theta(k));
                b = round(y - R * sin_theta(k));
                
                % Comprobar que el voto cae dentro de los límites de la matriz
                if a > 0 && a <= columnas && b > 0 && b <= filas
                    acumulador(b, a) = acumulador(b, a) + 1;
                end
            end
        end
        
    end
end

% 5. Encontrar MÚLTIPLES picos (centros de los círculos)

% Encontrar los picos locales en la matriz de acumulación
picos = imregionalmax(acumulador);

% Definir un umbral (ej. el pico debe tener al menos el 60% de los votos del pico máximo)
max_votos_global = max(acumulador(:));
umbral = 0.6 * max_votos_global; 

% Filtrar: Quedarnos solo con los picos locales que superen el umbral
picos_validos = picos & (acumulador >= umbral);

% Obtener las coordenadas (x, y) de todos los centros válidos encontrados
[centros_y, centros_x] = find(picos_validos);

% Mostrar resultados
imshow(I); hold on;
% Dibujar una cruz en cada centro detectado
plot(centros_x, centros_y, 'r+', 'MarkerSize', 10, 'LineWidth', 2);
% Dibujar las circunferencias
viscircles([centros_x, centros_y], R, 'Color', 'b');