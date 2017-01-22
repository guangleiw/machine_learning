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
[row,col] = size(X);
z=X*theta;
hz=sigmoid(z);

[row_theta,col_theta]=size(theta);
theta_part_1=theta(1,:);
theta_part_2=theta(2:row_theta,:);

J=((-1.*y)'*log(hz)+(y-1)'*log(1-hz))/(row)+theta_part_2'*theta_part_2*lambda/(2*row);

for j=1:(col)    
    grad(j)=(hz-y)'*X(:,j);
    if 1==j
        grad(1)=grad(j)/row;
        continue;
    end
    grad(j)=grad(j)/row + theta(j)*lambda/row;
end

% =============================================================

end
