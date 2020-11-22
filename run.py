from PIL import Image 
import os
import numpy
import copy
import random
import math

N = 4096	# No. of features or variables in the dataset
E = 3600	# No. of Principal Components to be chosen

if __name__ == "__main__":
	# Dataset is stored as an array of arrays of greyscale values
	input_dataset = []           # Initializing an empty array to store the input dataset
	os.chdir('My_Dataset/')
	for item in os.listdir(os.curdir):
		image = Image.open(item)
		input_dataset.append(list(image.getdata()))       # input_dataset now has the entire data as (30 x 4096) matrix

	# Copying the dataset and finding the covariance matrix of it
	normalized_dataset = copy.deepcopy(input_dataset)   	# Initializing a copy of the input dataset to perform actions without disturbing input dataset
	for i in range(N):
		mean = 0
		for j in range(len(normalized_dataset)):
			mean += normalized_dataset[j][i]
		mean /= len(normalized_dataset)			# Mean for the current feature is calculated
		for j in range(len(normalized_dataset)):
			normalized_dataset[j][i] -= mean 		# The mean is subtracted from respective feature values, as the first step of finding the covariance matrix of the dataset
		
	X = numpy.transpose(normalized_dataset).tolist() 
	Y = normalized_dataset
	covariance_matrix = numpy.dot(numpy.dot(X,Y),1/((len(normalized_dataset)-1))).tolist()	# covariance matrix is calculated from the normalized dataset and stored as a (4096 x 4096) matrix
	
	# Performing Eigen Value Decomposition on the covariance matrix and extracting principal components
	eigenvector_matrix = numpy.linalg.svd(covariance_matrix)[2].tolist()		# Library function used to do the singular value decomposition
	
	principal_components = []			# Initializing an empty array to store the principal components
	for row in eigenvector_matrix[:E]:		# Only the first E eigen vectors are considered
		principal_components.append([])
		norm = 0
		for element in row:
			norm += (element)**2
		norm = math.sqrt(norm)			# Norm for the current eigen vector is calculated
		for element in row:
			principal_components[-1].append(element/norm)		# principal_components now has the set of first E principal components as unit vectors of reals.

	# Reducing the dataset by transforming to the principal components' vector space and using that to reconstruct
	X = input_dataset
	Y = numpy.transpose(principal_components).tolist()
	reduced_dataset = numpy.dot(X,Y).tolist()		# reduced_dataset has the transformed data as (30 x E) matrix
	
	X = reduced_dataset
	Y = principal_components
	reconstructed_dataset = numpy.dot(X,Y).tolist()	# reconstructed_dataset has the reconstructed data as (30 x 4096) matrix
	
	# normalizing the reconstructed data into the range (0,255) to be able to reproduce images
	minimum = 0
	maximum = 255
	for i in range(30):
		for j in range(N):
			minimum = min(minimum, int(reconstructed_dataset[i][j]))
			maximum = max(maximum, int(reconstructed_dataset[i][j]))

	for i in range(30):
		for j in range(N):
			reconstructed_dataset[i][j] = int(255*(int(reconstructed_dataset[i][j]) - minimum)/(maximum - minimum))	# normalized the reconstructed data

	r = random.randint(0,29)
	reconstructed_image = Image.frombytes('L', (64,64), bytes(reconstructed_dataset[r]))	# Forming the reconstructed image of a randomly chosen picture
	reconstructed_image.show()
	original_image = Image.frombytes('L', (64,64), bytes(input_dataset[r]))
	original_image.show()

