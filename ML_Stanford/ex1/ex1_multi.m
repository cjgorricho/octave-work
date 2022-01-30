%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha_div = 3;
iter_mult = 2;
alpha = 1;
num_iters = 50;
max_loops = 5;

alpha_hist = zeros(max_loops);
iters_hist = zeros(max_loops);
theta = zeros(max_loops, 3);
J_all = NaN(max_loops, num_iters * iter_mult ^ max_loops);

for i = 1:max_loops

    % Init Theta and Run Gradient Descent 

    [theta(i,:), J_history] = gradientDescentMulti(X, y, theta(i, :)', alpha, num_iters);
    J_history = [J_history; NaN(size(J_all,2)-size(J_history), 1)];
    J_all(i, :) = J_history;
    
    alpha_hist(i) = alpha;
    iters_hist(i) = num_iters;
    
    if i == max_loops 
      break
    end
    
    alpha = alpha / alpha_div;
    num_iters = num_iters * iter_mult;

end

% Plot the convergence graph
figure;

xlabel('Number of iterations');
ylabel('Cost J');
hold on;

colors = ['r', 'g', 'y', 'k'];
axis ([0 200]);
legend_text = [];

for i = 1:max_loops-1
  
    plot(1:numel(J_all(i, :)), J_all(i, :), colors(i), 'LineWidth', 2);
    st_text = sprintf('Alpha: %.3f Iter: %d', alpha_hist(i), iters_hist(i));
    legend_text = [legend_text; st_text];

end

plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
st_text = sprintf('Alpha: %.3f Iter: %d', alpha, num_iters);
legend_text = [legend_text; st_text];
legend(legend_text, 'fontsize', 12);

hold off;

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta(max_loops, :)');
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
h_norm = ([1650 3]-mu)./sigma;
h_norm = [1 h_norm];

price = h_norm * theta(max_loops, :)'; % You should change this


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = [1 1650 3] * theta; % You should change this


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

