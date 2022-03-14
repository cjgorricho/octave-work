function visualizeBoundaryLinear(X, y, model, rows, cols, C, i)
%VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
%SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
%   learned by the SVM and overlays the data on it

w = model.w;
b = model.b;
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);

subplot(rows, cols, i)
    plotData(X, y);
    hold on;
    plot(xp, yp, '-b'); 
    hold off
    title(sprintf('Support Vector Machine\nSensitivity to C\n(C = %d)', C));
    xlabel('X');
    ylabel('y');
    legend('Positive', 'Negative', 'Linear boundary', 'location', 'northeast');


end
