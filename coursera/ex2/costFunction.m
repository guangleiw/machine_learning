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
[row,col] = size(X);
% ===========feature scaling==============
% mean_x = mean(X(:,2:col));
% max_x = max(X(:,2:col));
% min_x=min(X(:,2:col));
% sub = (bsxfun(@minus,X(:,2:col),mean_x));
% 
% ma_mi = repmat(max_x-min_x,row,1);
% X= [X(:,1) (sub./ma_mi)];
% ========================================
z = X*theta;
hz=sigmoid(z);

J=((-1.*y)'*log(hz)+(y-1)'*log(1-hz))/(row);

for j=1:(col)
    grad(j)=(hz-y)'*X(:,j);
    grad(j)=grad(j)/row;
end

% =============================================================

end
