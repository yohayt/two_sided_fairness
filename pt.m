

function [x,fval] = pt()
    fname='mnmxfair1';
    filename = strcat(strcat('temp/fa_', fname), '.csv');
    data = readmatrix(filename, 'HeaderLines', 1);
    
    % Extract parameters
    xvars = data(1);
    end_sums = data(2);
    erows = data(3);
    ierows = data(4);
    nf = data(5);
    nv = data(6);

    % read constraints
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
    mu = csvread(strcat(strcat('temp/Mu_', fname), '.csv'));

    x0 = zeros(size(A, 2),1);
    % generate non linear constraints from data
    handle = @(x) getNonLinearConstraints(x,mu, Ae(1:end_sums,:), xvars,nf);

    options = optimoptions(@fmincon,'Display', 'iter','MaxFunEvals',1000000, 'MaxIterations', 1000000, 'StepTolerance',1e-20);

    %run optimizer for the non-linear program
    [x, fval] = fmincon(@getLastElement,x0, A,b,Ae,be,[],[],handle, options);

end

function lastElement = getLastElement(vector)
    % Get the last element of the vector
    lastElement = vector(length(vector));
end

% generate non linear constraints
function [c, ce] = getNonLinearConstraints(x,mu, Ae, xvars, nf)
    [numRows, ~] = size(Ae);
    
    c = [];
    ce = [];
    rcols = xvars / numRows;

    for row = 1:numRows
        rowIndex = row;
        
        nonZeroIndices = find(Ae(rowIndex, :)); %available workers

        su = 0;

        for i = 1:length(nonZeroIndices)
            value = nonZeroIndices(i);
            ind = xvars+value - (row-1)*rcols;
            ind2 = xvars + nf + value - (row-1)*rcols;
            su = su + x(value) * mu(value) * x(ind2)/(1-x(ind));
        end
        su = su - x(xvars+rcols*2+row);
        su2 = x(xvars+rcols*2+row)-x(length(x));

        c = [c,su2];
        ce = [ce, su];
    end
end
