function [theta, J_history, theta_hist] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters+1, 1);
theta_hist = zeros(num_iters+1, 2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % Save the cost J in every iteration    
    
    theta_hist(iter,:) = theta';
    J_history(iter) = computeCost(X, y, theta);
    
    theta = theta - (alpha/m)*sum((X*theta-y).*X)';




    % ============================================================

    
    
end

theta_hist(num_iters+1,:)=theta';
J_history(num_iters+1) = computeCost(X, y, theta);
    


end
