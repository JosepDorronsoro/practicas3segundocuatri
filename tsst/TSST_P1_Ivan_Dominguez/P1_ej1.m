% Ejercicio 1.1

clear all;
close all;
clc;

% Definimos los dos polinomios P(z) y Q(z):

pz = [2 3 4 0];
qz = [3 4 5 6];

% Y este sería el polinomio de su producto:

pqz = conv(pz, qz);

% Guardamos los factores de normalización de P(z) y Q(z)

norm_p = 2;
norm_q = 3;

% Normalizamos los polinomios P y Q:

pz_norm = pz/norm_p;
qz_norm = qz/norm_q;

% Obtenemos sus raíces:

roots_pz = roots(pz_norm);
roots_qz = roots(qz_norm);

% Y ahora representamos en un vector de orden dos cada raíz:

% Para P(z):

root1_pz = [1 -roots_pz(1)];
root2_pz = [1 -roots_pz(2)];
root3_pz = [1 -roots_pz(3)];

% Para Q(z):

root1_qz = [1 -roots_qz(1)];
root2_qz = [1 -roots_qz(2)];
root3_qz = [1 -roots_qz(3)];

% Y ahora, haciendo su producto deberíamos obtener
% algo parecido a sus polinomios originales:

maybe_pz = norm_p * conv(conv(root1_pz, root2_pz), root3_pz);
maybe_qz = norm_q * conv(conv(root1_qz, root2_qz), root3_qz);

% Pasando por pantalla vemos claramente que son exactamente iguales. 