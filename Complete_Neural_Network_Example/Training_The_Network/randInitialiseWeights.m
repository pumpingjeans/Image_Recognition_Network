function W = randInitialiseWeights(L_in, L_out)
  
  #parameters given the dimensions of their corresponding layer
  W = zeros(L_out, 1+L_in);
  
  epsilon_init = 0.12;
  
  #parameters randomly generated between epsilion and -epsilon
  W = rand(L_out, 1+L_in)*(2*epsilon_init)-epsilon_init;

endfunction
