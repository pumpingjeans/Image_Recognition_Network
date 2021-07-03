import numpy as np
from PIL import Image
import os
import warnings

def Network(file_location):

	def sigmoid(x):
		return 1/(1+np.exp(-x))

	def dataConvert(image):
		image = Image.open(file_location, 'r')
		image = image.convert('L') ## makes image grayscale
		data = np.asarray(image)
		resized = np.resize(data,(100,100))
		input_data = np.reshape(resized,(10000,1))
		return input_data

	def forwardProp(input_layer, Theta1, Theta2):
		one = np.array([[1]])
		a0 = np.append(one, input_layer, axis=0)

		z1 = Theta1.dot(a0)
		a1 = np.array([sigmoid(xi) for xi in z1])
		a1 = np.append(one, a1, axis=0)

		z2 = Theta2.dot(a1)
		a2 = np.array([sigmoid(xi) for xi in z2])
		return a2

	def predict(output):
		MaxElement = np.amax(output)
		position = np.where(output == MaxElement)
		mushroom = int(position[0]-1)
		return mushroom

	#A tuple stores the names of my mushroom species with a fixed index
	fungi = ('Amanita muscaria','Cantharellus cibarius','Flammulina velutipes','Agaricus campestris','Grifola frondosa'\
		'Craterellus cornucopioides', 'Morchella elata', 'Pleurotus ostreatus', 'Boletus edulis', 'Lentinula edodes')


	#changing the default directory
	os.chdir("C:/users/Dale/Downloads/Python CNN")

	#to ignore the overflow warning in the exp function *used in the sigmoid function
	warnings.filterwarnings("ignore")

	#file_location = input("File location of your image: ")

	#Loading the precalculated parameters
	Theta1 = np.loadtxt("paramsLayer1.csv", delimiter=",")
	Theta2 = np.loadtxt("paramsLayer2.csv", delimiter=",")

	#loading the input data

	#applying the neural network to the data
	input_layer = dataConvert(file_location)
	output = forwardProp(input_layer, Theta1, Theta2)
	prediction = predict(output)
	prediction = fungi[prediction]
	#displaying the prediction
	#print(prediction)
	return prediction



	#This codes for my most basic Neural Network with 1 hidden layer of 100 units

