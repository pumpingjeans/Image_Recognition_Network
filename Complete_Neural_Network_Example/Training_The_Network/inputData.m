function [X, Y, labels] = inputData(p)
  m = size(p, 1);                   #m is the number of training examples
  labels = p(:, end);
  
  X = p(:,1:end-1);                 #X is the image data, without labels
  Y = zeros(m, max(p(:,end)));      #Y starts as a zeros matrix with m rows
                                    #and a column for each species
  for k = 1:m
    for i = 1:max(p(:,end))
      if p(k,end) == i              #Depending on the number assigned to the species
        Y(k,i) = 1;                 #a corresponding binary value is given
      end;                          #with a one in the location the number specifies
    end;                            #e.g. 10 could be [0000000001]
  end;                              #and 3 could be [0010000000]
      

endfunction

#Code is designed to change depending on the size of the input, not set arguments

#This code splits the data into input and output data, converting the integer labels
#to binary row vectors