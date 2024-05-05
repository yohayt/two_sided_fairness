function [x,fval] = ps()

fname = 'mnmxfair1';
filename = strcat(strcat('temp/fa_', fname), '.csv');
data = readmatrix(filename, 'HeaderLines', 1);

% Extract parameters
nf = data(1);
nv = data(2);
erows = data(3);
ierows = data(4);
all = data(5);

% read objective and constraints
val = csvread(strcat(strcat('temp/vals_', fname), '.csv'));
I = csvread(strcat(strcat('temp/I_', fname), '.csv'))+1;
J = csvread(strcat(strcat('temp/J_', fname), '.csv'))+1;
b = csvread(strcat(strcat('temp/b_', fname), '.csv'));
A = sparse(I,J, val, ierows,nv);

vale = csvread(strcat(strcat('temp/valse_', fname), '.csv'));
Ie = csvread(strcat(strcat('temp/Ie_', fname), '.csv'))+1;
Je = csvread(strcat(strcat('temp/Je_', fname), '.csv'))+1;
be = csvread(strcat(strcat('temp/be_', fname), '.csv'));
Ae = sparse(Ie,Je, vale, erows,nv);
c = zeros(1, nv);
c(all+1) = 1;

options = optimoptions(@linprog,'Display', 'iter');

%run linear program
[x, fval] = linprog(c,A,b,Ae,be,[],[], options);

end

