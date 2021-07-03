function g = sigmoid(z)
  #sigmoid computes the sigmoid function
  g = 1.0 ./(1.0 + exp(-z));  

endfunction
