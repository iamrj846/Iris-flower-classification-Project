function [J, grad] = costFunction(theta, X, y, lambda)
%COSTFUNCTION Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


J = ((sum((log(sigmoid(X*theta)))' * ((-1)*y))-sum((log(1 - sigmoid(X*theta)))' * (1 - y)))/m); 
regularized_term = (sum((theta(2:length(theta), 1)).^2)) * lambda/(2*m); 
for i=1:length(J),
  J(i) = J(i) + regularized_term;
endfor
grad = (X' * (sigmoid(X*theta) - y))/m;
for i = 2:length(grad),
  grad(i) = grad(i) + (lambda/m) * theta(i);
endfor


% =============================================================

end
