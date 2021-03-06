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

% initialize variables;

sigma1 = 0;
sigma2 = 0;
hypothesis = zeros(m, 1);


% calculate cost;
hypothesis = X * theta;

sigma1 = ((hypothesis-y)') * (hypothesis-y);
sigma2 = (theta(2:end)') * theta(2:end);

J = (1/(2*m))*sigma1 + (lambda/(2*m))*sigma2;

% ===================Regularized linear regression gradient===========================
temp = [0;theta(2:end)];
grad = (1/m)*(X'*(hypothesis-y)) + (lambda/m)*temp;

grad = grad(:);

end
