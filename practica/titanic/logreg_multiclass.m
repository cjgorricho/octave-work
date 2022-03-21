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

%% Setup the parameters you will use for this part of the exercise
num_labels = 6;           % 6 labels corresponding to combinations of Class and Sex

%% Load Data
%  xlsread will be used to load data directly from Excel
%  Col  1: Passenger ID (not used)
%  Col  2: original label y (survived: 1, did not survive:0)
%  Col  3: Passenger Class (PClass: 1, 2, 3)
%  Col  4: Name (not used)
%  Col  5: Sex_txt (male, female)
%  Col  6: Sex_num (male: 1, female: 2) 
%  Col  7: Age (not used because it has missing values)
%  Col  8: Age_adj; for training purposes average was calculated by survived/class/sex
%  Col  9: SibSp - # of siblings or spouse 
%  Col 10: Parch - # of parents or children
%  Col 11: Ticket (not used)
%  Col 12: Fare (not used)
%  Col 13: Cabin (not used)
%  Col 14: Embarked_txt (C, S, Q)
%  Col 15: Embarked_num (C:1, S: 2, Q:3)
%  Col 16: Class_sex; combination of Class and Sex. These are the classes
%  col 17 - 22: y vectors for each class in one-vs-all algorythm

% Load data - combination of Class, Sex, SibSp, Parch, Embarked
data = csvread('train.csv'); % Read file and worksheet
X = data(3:end, [3, 4, 6, 7]); % Load data according to dictionary above except 
y = data(3:end, 10:21);

% Load data - Passenger Age and add npols polinomials. If commented (%) not used
%X_age = data(4:end, 8);
%npols = 2;
%X_age_poly = polyFeatures(X_age, npols);

% Integreate data
m = size(X,1);
%X = [ones(m,1), X, X_age_poly]; % Use only if age is a feature
X = [ones(m,1), X]; % Use only if age is NOT a feature

% Create random training and validation sets
rand_ind = randperm(m);
train_perc = 0.7;
train_ind = round(train_perc * m);
X_train = X(rand_ind(1:train_ind), :); 
y_train = y(rand_ind(1:train_ind));
X_val = X(rand_ind(train_ind+1:end), :); 
y_val = y(rand_ind(train_ind+1:end));

% Initialize fitting parameters (DONE IN ONEVSALL FUNCTION)
%initial_theta = rand(size(X, 2), 1); 
initial_theta = zeros(size(X, 2), size(y,2)); 
%initial_theta = ones(size(X, 2), 1); 

% Set regularization parameter lambda and theta
lambda = 2.75;

% Set Options
options = optimset('MaxIter', 400, 'GradObj', 'on');

% Optimize
costFunction = @(t)(costFunctionReg(t, X_train, y_train, lambda));

[theta, J, exit_flag, output, grad] = ...
	fminunc(costFunction, initial_theta, options);

% Calculate predicted values with theta and print accuracy results
p = predict(theta, X_val);
fprintf('Train Accuracy: %f\n', mean(double(p == y_val)) * 100);

% Plot Learning Curve using larningCurve_rand.m adjusted for this exercise
%[error_train, error_val] = ...
%  learningCurve_rand(X_train, y_train, X_val, y_val, lambda);

%close all;
%X_plot = 1:size(error_train, 1);
%X_plot = X_plot';
%y_plot = [error_train, error_val];
%figure;
%plot(X_plot, y_plot);
%legend('Train', 'Cross Validation', 'location', 'northeast');
%axis([0 100 0 10]);
