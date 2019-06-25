function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

matrix = theta;
n = length(theta(1,:));
matrix(1, :) = zeros(1, n);

J = 1/(2*m) * (sum((X * theta - y) .^ 2)  + lambda * matrix' * matrix);
grad = 1/m * ((X * theta - y)' * X)' + lambda/m * matrix;








% =========================================================================

grad = grad(:);

end
