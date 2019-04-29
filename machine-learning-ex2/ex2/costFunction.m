function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

sigma = 0;
hypothesis = zeros(m, 1);

for i = 1:m
    hypothesis(i) = sigmoid(dot(theta, X(i,:)));
endfor

for i = 1:m
    each = -y(i)*log(hypothesis(i)) - (1-y(i))*log(1-hypothesis(i));
    sigma = sigma + each;
endfor

J = sigma / m;

for i = 1:size(grad)
    grad(i) = dot(hypothesis-y, X(:,i)) / m;
endfor

% =============================================================

end
