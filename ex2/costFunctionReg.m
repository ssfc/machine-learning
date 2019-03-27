function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sigma1 = 0;
sigma2 = 0;
sigma3 = 0;
hypothesis = zeros(m, 1);

for i = 1:m
    hypothesis(i) = sigmoid(dot(theta, X(i,:)));
endfor

for i = 1:m
    each = -y(i)*log(hypothesis(i)) - (1-y(i))*log(1-hypothesis(i));
    sigma1 = sigma1 + each;
endfor

for i = 2:size(theta)
    sigma2 = sigma2 + (theta(i))^2;
endfor

J = (1/m)*sigma1 + (lambda/(2*m))*sigma2;

% compute the partial derivatives and set grad to the partial ===
% derivatives of the cost w.r.t. each parameter in theta

grad(1) = (1/m) * dot(hypothesis-y, X(1:end, 1));


% grad(2) = (1/m) * dot(hypothesis-y, X(1:end, 2)) + (lambda / m)*theta(2);

for i = 2:size(theta)
    grad(i) = (1/m) * dot(hypothesis-y, X(1:end, i)) + (lambda / m)*theta(i);
endfor



% =============================================================

end
