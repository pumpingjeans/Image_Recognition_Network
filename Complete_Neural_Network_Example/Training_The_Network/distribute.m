function [trainData,testData] = distribute(p)
  m = size(p, 1);
  
  shuffled = p(randperm(size(p,1)), :);          #ordered data is shuffled for randomisation
  
  trainSize = round(m*0.8);                      #calculate the number of train examples
  testSize = round(m*0.2);                       #calculate the number of test examples
  
  
  trainData = shuffled(1:trainSize, :);          #training data taken from the input data
  testData = shuffled((trainSize+1):end, :);     #testing data taken from the input data

endfunction
