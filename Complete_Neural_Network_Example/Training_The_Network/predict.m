function p = predict(Theta1, Theta2, Theta3, X)
  
  #m is the number of samples
  m = size(X, 1);
  p = zeros(m, 1);
  
  #forward propogate throught the network
  h1 = sigmoid([ones(m, 1) X] * Theta1');
  h2 = sigmoid([ones(m, 1) h1] * Theta2');
  h3 = sigmoid([ones(m, 1) h2] * Theta3');
  
  #finds the indexes of the the maximum values of h2 and returns
  #them in the output vector p
  [dummy, p] = max(h3, [], 2);
  #What it does is find the integer label of the node with highest probability
  #in a sample
  #It does this for all the samples until it has an m*1 vector of integer labels

endfunction
