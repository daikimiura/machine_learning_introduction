function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

 sums = zeros(columns(X), 1);
  for j = 1:columns(X)
  	for i = 1:m
  		 sums(j) = sums(j) + (theta' * X(i, :)' - y(i)) * X(i, j);
 	end
  end

  for j = 1:columns(X)
	theta(j) = theta(j) - alpha * (1/m) * sums(j);
  end

	J = computeCostMulti(X, y, theta)

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
