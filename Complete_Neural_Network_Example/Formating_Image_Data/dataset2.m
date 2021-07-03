function dataset2
  
  A = glob('Amanita_Muscaria\*.jpg');         #This code locates the file location of the images
  B = glob('Chanterelle\*.jpg');              #and stores the image file locations in arrays
  C = glob('Enokitake\*.jpg');                #Each species is assigned to their unique variable
  D = glob('Field_Mushroom\*.jpg');
  E = glob('Hen_of_the_Woods\*.jpg');
  F = glob('Horn_of_Plenty\*.jpg');
  G = glob('Morel\*.jpg');
  H = glob('Oyster\*.jpg');
  I = glob('Penny_Bun\*.jpg');
  J = glob('Shiitake\*.jpg');
                                                        #I created a cellular array and normal array because they both have specific
                                                        #attributes which I will nead later on
  names = [A; B; C; D; E; F; G; H; I; J];               #File names of the images stored in an array
  names2 = {A; B; C; D; E; F; G; H; I; J};              #File names of the images stored in a cellular array
  imagelist = zeros(length(names), 10001);              #A matrix of zeros with the dimesnsions of the output is created

                              
  for k = 1:length(names)                               #For loop loops through all the mushroom file names
	  image = imread(names{k, 1});                        #Convert the filename to image data
    check = names{k, 1};    
    
    if size(image, 3) == 3                              #Checks if image is RGB 
	    GrayImage = rgb2gray(image);                      #If it is, it converts it to grayscale
      for i = 1:length(names2)                          #This code assigns a number to each image
        if ismember(check, names2{i})                   #This label is stored in the last column 
          imagelist(k, 10001) = i;                    
        end;                                            
      end;                                              
                                                        
    elseif size(image, 3) == 1                          #If the image is already grayscale                          
      GrayImage = image;                                #it is kept in the same format
      for i = 1:length(names2)                          #Again, this code lables the images
        if ismember(check, names2{i})                    
          imagelist(k, 10001) = i;       
        end;                                            
      end;                                              
        
    else                                                #If the image is neither RGB or grayscale format
      bin = image;                                      #it is removed
    end                                                   
                                                    
	  SqrImage = imresize(GrayImage, [100, 100]);         #The image is resized to be 100*100 pixels
	  UnrlImage = reshape(SqrImage, 1, 10000);            #The resized image is unrolled into a row vector of 10000 columns  
	  imagelist(k, 1:10000) = UnrlImage;                  #The unrolled image is stored in columns 1 to 10000 of the matrix imagelist
  end;
  dlmwrite('mushroomdata3.txt', imagelist)               #The matrix of images 'imagelist' is uploaded as a .txt file    
endfunction                                             #(with the same format as a csv file)


 #Creating the code for labeling the images with numbers was very hard
 #I had to be very specific in how used cells and arrays to accomplish my goal
 #Optimising this section of code was by far the most challenging part of coding this function
 
 