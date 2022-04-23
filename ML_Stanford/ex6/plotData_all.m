function plotData_all(X, y, Xval, yval)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
%
% Note: This was slightly modified such that it expects y = 1 or y = 0

% Find Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);
pos_val = find(yval == 1); neg_val = find(yval == 0);

% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 0.5, 'MarkerSize', 5)
hold on;
plot(X(neg, 1), X(neg, 2), 'ko', 'LineWidth', 0.5, 'MarkerFaceColor', 'y', 'MarkerSize', 5)
plot(Xval(pos_val, 1), Xval(pos_val, 2), 'b+',  'LineWidth', 0.5, 'MarkerSize', 5)
plot(Xval(neg_val, 1), Xval(neg_val, 2), 'ko', 'MarkerFaceColor', 'r', 'LineWidth', 0.5, 'MarkerSize', 5)
hold off;
legend('y=1', 'y=0', 'yval=1', 'yval=0');

end
