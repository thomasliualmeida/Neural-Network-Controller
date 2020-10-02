clear all, close all, clc

m = 0.5;
L = 1;
g = 9.8;

A = [0 1; 
    (sqrt(2)*g)/(2*L) 0];
B = [0;
    1/(m*L^2)];
%check whether it is controllable. Rank of the control. matrix must be 2
rank(ctrb(A,B))

%% pole placement and LQR
Q = [100 0
     0 10];
R = 0.1;

K = lqr(A, B, Q ,R)


