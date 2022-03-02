function [all_theta] = normalEqn(X, y, lambda)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

% Calculate L
L = eye(size(X, 2));
L(1,1) = 0;


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

all_theta = inv(X'*X+lambda*L)*X'*y;


% -------------------------------------------------------------


% ============================================================

end
