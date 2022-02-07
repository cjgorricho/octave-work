%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction_vec.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');
m = size(X, 1);


%% ================ Part 2: Loading Parameters ================
% In this part of the exercise, we load some pre-initialized 
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];


%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.


%  You should also try different values of lambda
lambda = 3;

lambda_div = 3;
iter_add = 0;
num_iters = 200;
max_loops = 5;

lambda_hist = zeros(1, max_loops + 1);
iters_hist = zeros(1, max_loops + 1);
time_hist = zeros(1, max_loops + 1);
accu_hist = zeros(1, max_loops + 1);
cost_all = NaN(max_loops, num_iters + iter_add * max_loops);

for i = 1:max_loops + 1

    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction_vec(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X, y, lambda);
    
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    
    options = optimset('MaxIter', num_iters);

    tic();
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    time_elapsed = toc();
    
    cost = [cost; NaN(size(cost_all,2)-size(cost), 1)]; % Complete cost vector with NaN
    cost_all(i, :) = cost;
    
    lambda_hist(i) = lambda;
    iters_hist(i) = num_iters;
    time_hist(i) = time_elapsed;
    
    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    
    fprintf('\nTime elapsed: %.2f minutes \n', time_elapsed / 60);

    
    %% ================= Part 10: Implement Predict =================
    %  After training the neural network, we would like to use it to predict
    %  the labels. You will now implement the "predict" function to use the
    %  neural network to predict the labels of the training set. This lets
    %  you compute the training set accuracy.
    
    pred = predict(Theta1, Theta2, X);
    accu_hist(i) = mean(double(pred == y)) * 100;
  
    fprintf('\nTraining Set Accuracy: %f\n', accu_hist(i));

    if i == max_loops + 1
      break
    end
    
    lambda = lambda / lambda_div;
    num_iters = num_iters + iter_add;
    
end


colors = ['r', 'g', 'm', 'k', 'c', 'b'];
%axis ([0 100]);
legend_text = [];

% Plot the convergence graphs
figure;
xlabel('Number of iterations');
ylabel('Cost J');
hold on;

for i = 1:max_loops + 1
  
    st_text = sprintf('Lambda: %.3f | Iter: %d | J min: %.3f', ... 
      lambda_hist(i), iters_hist(i), min(cost_all(i, :)));
    legend_text = [legend_text; st_text];
    plot(1:numel(cost_all(i,:)), cost_all(i,:), colors(i));
    
end

legend(legend_text, 'fontsize', 10);
hold off;

figure;
xlabel('Time to reach solution');
ylabel('Accuracy');
hold on;
legend_text = [];



for i = 1:max_loops + 1
  
    st_text = sprintf('Lambda: %.3f | Iter: %d | Time: %.1f sec | Acc: %.2f%%', ... 
      lambda_hist(i), iters_hist(i), time_hist(i), accu_hist(i));
    legend_text = [legend_text; st_text];
    plot(time_hist(i), accu_hist(i), 'marker', 'x', 'linewidth', 2, colors(i));
    
end

legend(legend_text, 'fontsize', 10, 'location', 'southeast');
hold off;
