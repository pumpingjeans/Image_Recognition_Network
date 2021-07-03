function [J, grad] = costFunction4(unrl_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   hidden_layer2_size, ...
                                   X, Y, lambda)
  
  num_labels = size(Y, 2);
  
  
  #resize the vector of unrolled parameters to their original dimensions
  Theta1Size = hidden_layer_size * (input_layer_size + 1);
  Theta2Size = hidden_layer2_size * (hidden_layer_size + 1);
  
  
  
  
  Theta1 = reshape(unrl_params(1:Theta1Size), hidden_layer_size, (input_layer_size + 1));
  
  Theta2 = reshape(unrl_params(1 + Theta1Size:(Theta1Size + Theta2Size)), ...
                   hidden_layer2_size, (hidden_layer_size + 1));
  
  Theta3 = reshape(unrl_params(1 + Theta1Size + Theta2Size:end), ...
                   num_labels, (hidden_layer2_size + 1));

  m = size(X, 1);
  J = 0;

  #Vectorised forward propogation:
  #=> forward propogation for all examples executed simulataneously
  a0 = [ones(m,1), X];
  
  z1 = a0 * Theta1';
  a1 = sigmoid(z1);
  a1 = [ones(m,1), a1];
  
  z2 = a1 * Theta2';
  a2 = sigmoid(z2);
  a2 = [ones(m,1), a2];
  
  z3 = a2 * Theta3';
  a3 = sigmoid(z3);
  
  #Cost function:
  #J = 1/m(-y*log(h*x) - (1-y)log(1-h*x)) + (lambda/2m)(theta1^2+ theta2^2)
  
  #J = 1/m(-y*log(h*x) - (1-y)log(1-h*x))
  J = sum(sum( -Y.*log(a3) - (1-Y).*log(1-a3) ))/m;
  
  #Add the regularisation pentalty
  #J = J +  (lambda/2m)(theta1^2+ theta2^2)
  J = J + (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) + sum(sum(Theta3(:,2:end).^2 )))* lambda/(2*m);  

  
  ##Here onwards is for calculating the partial derivatives of the cost function
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  Theta3_grad = zeros(size(Theta3));
 
  #sample by sample
  for i = 1:m
    #Forward propogate
    a_0 = X(i, :)';
    a_0 = [1; a_0];
    
    z_1 = Theta1 * a_0;
    a_1 = sigmoid(z_1);
    a_1 = [1; a_1];
    
    z_2 = Theta2 * a_1;
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2];
    
    z_3 = Theta3 *a_2;
    a_3 = sigmoid(z_3);
    
    #Error
    Ycurrent = Y(i, :)';
    delta_3 = a_3 - Ycurrent;
    
    #Propogate error backwards
    delta_2 = (Theta3' * delta_3).* sigmoidGradient([1; z_2]);
    delta_2 = delta_2(2:end);
    delta_1 = (Theta2' * delta_2).* sigmoidGradient([1; z_1]);
    delta_1 = delta_1(2:end);
    
    dt3 = delta_3 * a_2';
    dt2 = delta_2 * a_1';
    dt1 = delta_1 * a_0';
    
    #The deltas are the error in the activation units
    #The dts are the partial derivatives of the cost function/error in parameters
    
    #The partial derivatives of each training example are added to the partial
    #derivatives of their previous example
    Theta3_grad = Theta3_grad + dt3;
    Theta2_grad = Theta2_grad + dt2;
    Theta1_grad = Theta1_grad + dt1;
  end;
  
  #divide by the number of training examples to get the average of the partial derivatives
  Theta1_grad = (1/m) * Theta1_grad;
  Theta2_grad = (1/m) * Theta2_grad;
  Theta3_grad = (1/m) * Theta3_grad;
  
  #Add the regularisation terms
  Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
  Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);
  Theta3_grad(:,2:end) = Theta3_grad(:,2:end) + (lambda/m) * Theta3(:,2:end);
  #Similar to the regularisation in the cost function, ignore the bias units
  #because they have no impact on the regularisation
  
  #grad is the unrolled vector of the the cost function partial derivatives
  grad = [Theta1_grad(:); Theta2_grad(:); Theta3_grad(:)];
endfunction
