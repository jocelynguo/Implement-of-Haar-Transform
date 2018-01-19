# import the python package 
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from numpy import matlib
from pywt import wavedec
import math

# define a function to covert the image to a gray scale image
def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
 
# define a function to get the proper Haar matrix and permutation matrix
def GetHaarMatrices(N):
	Q = np.matrix("[1,1;1,-1]")
	M = int(N/2)
	T = np.kron(matlib.eye(M),Q)/np.sqrt(2)
	P = np.vstack((matlib.eye(N)[::2,:],matlib.eye(N)[1::2,:]))
	return T,P

# reads in a jpeg image
# answer of Q1a
A = imread('image.jpg')

# show the jpeg image in a figure
plt.imshow(A, cmap = plt.get_cmap('gray'))
plt.show()

# represent the jpeg image(before apply gray scale function) as a 256 by 256 matrix
# answer of Q1c
A = imresize(A, [256, 256], interp = 'bicubic')

# Apply the rgb2gray function to the jpeg image
A = rgb2gray(A)

# show the jpeg image in a figure
# answer of Q1b
plt.imshow(A, cmap = plt.get_cmap('gray'))
plt.show()

# resize the gray scale jpeg image in a 256 by 256 matrix
# answer of Q2
A = imresize(A, [256, 256], interp = 'bicubic')
print(A)

# show the resize jpeg image
# answer of Q4
plt.imshow(A, interpolation='nearest')
plt.show()

# apply the GetHaarMatrices function to get the Haar and permutation matrices
T,P = GetHaarMatrices(256)

# implementing the forward 2D Haar transform of a matrix
B = P*T*A*T.T*P.T

# show the image after the forward 2D Haar transform
# answer of Q3(1)
plt.imshow(B, interpolation='nearest')
plt.show()

# implementing the inverse 2D Haar transform of a matrix
inverseB = T.T*P.T*B*P*T

# show the image after the inverse 2D Haar transform of B
# answer of Q3(2)
plt.imshow(inverseB, interpolation='nearest')
plt.show()

# declare a signal vector s
s = np.array([2, 7, 1, 8, 2, 8, 1, 8])

# check how many levels of Haar transform need to full-level processing of s
max_Level = int(math.log(len(s), 2))

# get the transpose of s
s = s[np.newaxis, : ].T

# 1-level of s
# get the proper Haar matrix and permutation matrix to start
T,P = GetHaarMatrices(len(s))
# get the result of s after 1-level of processing
s = np.array(P*T*s)
# prepare the next level of processing which won't make change of the second half
s1 = s[:len(s)/2]
# save the second half after 1-level for future concatenate
D1 = s[len(s)/2:len(s)]

# 2-level processing of s
# get the proper Haar matrix and permutation matrix to start
T,P = GetHaarMatrices(len(s1))
# get the result of s after 2-level of processing
s2 = np.array(P*T*s1)
# prepare the next level of processing which won't make change of the second half
s3 = s2[:len(s2)/2]
# save the second half after 2-level for future concatenate
D2 = s2[len(s2)/2:len(s2)]

# 3-level processing of s
# get the proper Haar matrix and permutation matrix to start
T,P = GetHaarMatrices(len(s3))
# get the result of s after 3-level of processing
s4 = np.array(P*T*s3)
# prepare the next level of processing which won't make change of the second half
s5 = s4[:len(s4)/2]
# save the second half after 3-level for future concatenate
D3 = s4[len(s4)/2:len(s4)]

# since the max_Level is 3, then concatenate the result together to get full-processing result of s
s_Haar = np.concatenate((s5, D3, D2, D1), axis=0)

# print the forward Haar transform of vector s
print(s_Haar)
