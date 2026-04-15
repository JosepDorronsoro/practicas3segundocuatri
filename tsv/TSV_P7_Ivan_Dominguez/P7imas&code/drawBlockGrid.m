function drawBlockGrid(blocks, color, lineWidth)
% drawBlockGrid  Dibuja los bordes de bloques rectangulares generados por split & merge.
%
% blocks: celda donde cada elemento es [fila, col, alto, ancho]
% color:  color de la línea, ej. 'r' o [1 0 0]
% lineWidth: grosor de línea

if nargin < 2
    color = 'r';
end
if nargin < 3
    lineWidth = 1.5;
end

hold on;
for k = 1:length(blocks)
    b = blocks{k};
    r = b(1);
    c = b(2);
    h = b(3);
    w = b(4);

    rectangle('Position', [c, r, w, h], ...
              'EdgeColor', color, ...
              'LineWidth', lineWidth);
end
hold off;

end