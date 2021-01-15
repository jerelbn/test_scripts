% Symbolic calculation of Euclidean Homography scale factor
clear all
close all

syms m11 m22 m33 m12 m13 m23
syms lam

% M = [m11 m12 m13
%      m12 m22 m23
%      m13 m23 m33];
% I = eye(3);
% 
% eq = simplify(det(M-lam*I))

H_hat = [ 0.806987   0.0820363 -0.143887
         -0.0645066  0.817627   0.00110625
         -0.163253   0.0824482  0.959433];
M = H_hat' * H_hat

m11 = M(1,1);
m22 = M(2,2);
m33 = M(3,3);
m12 = M(1,2);
m13 = M(1,3);
m23 = M(2,3);

a2 = -(m11 + m22 + m33)
a1 = m11*m22 + m11*m33 + m22*m33 - m12*m12 - m13*m13 - m23*m23
a0 = m12*m12*m33 + m13*m13*m22 + m23*m23*m11 - m11*m22*m33 - 2*m12*m13*m23

% a2 = -2.3053;
% a1 = 1.67;
% a0 = -0.383899;

eq1 = lam^3+a2*lam^2+a1*lam+a0;
sol = solve(eq1,lam)