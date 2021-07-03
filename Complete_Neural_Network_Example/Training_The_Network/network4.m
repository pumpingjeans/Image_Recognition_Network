function network4

  #read the data
  ordered = dlmread("mushroomdata2.txt");
  
  #Shuffle and split the data into training and test data
  [trainData, testData] = distribute(ordered);
  
  #Seperate the training and test data into input and output data
  [X1,Y1,labels1] = inputData(trainData);
  [X2,Y2,labels2] = inputData(testData);
  
  #Creating the dimensions of the network
  xSize = size(X1, 2);
  a1Size = 1000;
  a2Size = 1000;
  ySize = size(Y1, 2);
  
  #randomly generating the parameters
  h = @randInitialiseWeights;
  initial_Theta1 = h(xSize, a1Size);
  initial_Theta2 = h(a1Size, a2Size);
  initial_Theta3 = h(a2Size, ySize);
  
  #Setting variables for use in fmincg
  initial_unrl_params = [initial_Theta1(:); initial_Theta2(:); initial_Theta3(:)];
  lambda = 1;
  options = optimset('MaxIter', 50);
  
  #create "short hand" for the cost function to be minimized  
  costF = @(p) costFunction4(p, xSize, a1Size, a2Size, X1, Y1, lambda);
  
  [unrl_params, cost] = fmincg(costF, initial_unrl_params, options);
  fprintf('The costs values');
  cost
  plot(cost);
  title('Cost versus number of iterations');
  xlabel('Number of iterations');
  ylabel('Cost');
  legend('Cost Function');
  
  #reshaping the unrolled recalculated parameters into their original dimensions
  Theta1Size = a1Size * (xSize+1);
  Theta2Size = a2Size * (a1Size+1);
  
  Theta1 = reshape(unrl_params(1:Theta1Size), a1Size, (xSize+1));
  Theta2 = reshape(unrl_params(1 + Theta1Size:(Theta1Size + Theta2Size)), a2Size, (a1Size+1));
  Theta3 = reshape(unrl_params(1 + Theta1Size + Theta2Size:end), ySize, (a2Size+1));
  
  #Get the predicted values of the network with the recalculated parameters
  #for the training and test data
  pred1 = predict(Theta1, Theta2, Theta3, X1);
  pred2 = predict(Theta1, Theta2, Theta3, X2);
  
  #These for loops track how many of the networks predictions are right in
  #the sum variables
  sum1 = 0;
  sum2 = 0;
  for i = 1:length(pred1)
    if pred1(i) == labels1(i)
      sum1 += 1;
    end
  end
  
  for i = 1:length(pred2)
    if pred2(i) == labels2(i)
      sum2 += 1;
    end
  end
  
  #The % accuracy of the network on the training and test data are calculated 
  accuracy1 = sum1/length(pred1) * 100;
  accuracy2 = sum2/length(pred2) * 100;

  #the %.f is the format function which maps the value I give into the string
  #the %% is actually just % in the printed output
  #the \n creates a new line
  fprintf("\nTraining Set Accuracy: %.f%%\n", accuracy1);
  fprintf("\nTesting Set Accuracy: %.f%%\n", accuracy2);
  
  #alternate method of displaying accuracy
  fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred1 == labels1)) * 100);
  fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred2 == labels2)) * 100);
  
endfunction
