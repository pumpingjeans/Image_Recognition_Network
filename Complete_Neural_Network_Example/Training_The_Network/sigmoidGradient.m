function g = sigmoidGradient(z)
  
  g = zeros(size(z));
  
  g = sigmoid(z).*(1-sigmoid(z));

endfunction

#function calculates the gradient of the sigmoid function
#which can by found through differentiation