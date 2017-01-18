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
% disp(theta);
% disp(X);
% disp(y);
z = X*theta;
hz=sigmoid(z);
disp(hz);
[row,col] = size(X);
p1=(-1.*y)'*log10(hz);
p2=(y-1)'*log10(1-hz);
J=(p1+p2)/(row);

for j=1:(col)
    grad(j)=(hz-y)'*X(:,j);
    grad(j)=grad(j)/row;
end


% =============================================================

end
