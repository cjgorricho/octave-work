%% Machine Learning Practice - Titanic: Logistic Regression
%
%  By Carlos Gorricho
%  cgorricho@heptagongroup.co
%
%  Description
%  ------------
%
%  This file contains code to solve Kaggle's Titanic challenge. There are 
%  a couple of Titanic datasets out there, but Kaggle's has some data fields
%  that make you analyze the problem from inception.
%  
%  This solution is based on a ML Logistic Regression algorithm.
%  I wil experiment with varios combinations of features, until I feel
%  comfortable with the solution.
%
%  In the end I will submit the solution to Kaggle to validate it.
%

%% Initialization
clear; close all; clc

%% Load Data
%  The original file is transformed to obtain the following file structure
%  Col 1: passenger consecutive ID, which is not used in the model
%  Col 2: label (y)
%  Col 3: Passenger Class (PClass: 1, 2, 3)
%  Col 4: Sex (male: 1, female: 2)
%  Col 5: Age; where data was missing I put the average for the class, sex and 
%         survived/not-survived. This should reduce the drag
%  Col 6: 
%  Col 7:
%  Col 8: Boarding Port (C: 1, S: 2, Q:3)

data = load('train_val.csv');
X = data(:, [3, 4, 5, 6]); y = data(:, 2);

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 0.1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
