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

% Load data
data = load('train_val.csv');
X = data(:, [3, 4, 5]); y = data(:, 2);
m = size(X,1);
X = [ones(m,1) X];


% Create random training and validation examples
rand_ind = randperm(m);
train_perc = 0.7;
train_ind = round(train_perc * m);
X_train = X(rand_ind(1:train_ind), :); y_train = y(rand_ind(1:train_ind));
X_val = X(rand_ind(train_ind+1:end), :); y_val = y(rand_ind(train_ind+1:end));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 3.5;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag, output, grad] = ...
	fminunc(@(t)(costFunctionReg(initial_theta, X_train, y_train, lambda)), options);

p = predict(theta, X_val);

fprintf('Train Accuracy: %f\n', mean(double(p == y_val)) * 100);
