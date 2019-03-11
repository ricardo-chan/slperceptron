from __future__ import print_function
import matplotlib, sys
from matplotlib import pyplot as plt
import numpy as np

def classifyTrainingPoint(trainingPoints, line):
	#	Checking to see which classification do the training points
	#	fall under.
	slope = line[0]
	b = line[1]

	for trainingPoint in trainingPoints:
		x1_Given = trainingPoint[1]
		x2_Given = trainingPoint[2]
		x2_Calculated = (x1_Given * slope) + b
		if x2_Calculated < x2_Given:
			trainingPoint[3] = 1.0
		else:
			trainingPoint[3] = 0.0

def genLine():
	x1, y1 = np.random.uniform(-1, 1, 2)	#	First point
	x2, y2 = np.random.uniform(-1, 1, 2)	#	Second point

	#	Taking the form of y = mx + b... Calculating.
	m = (y2 - y1) / (x2 - x1)
	b = y1 - (m * x1)

	#	Plotting line
	#hor = np.linspace(-1,1)
	#xs = [x1, x2]
	#ys = [y1, y2]
	#c1s = plt.scatter(xs,ys,s=40.0,c='g')
	#plt.plot(hor,m*hor + b, label='Division')

	y = [m, b]
	return y

def genData(trainingPoints = 10):
	dataSet = []
	for i in range(trainingPoints):
		trainingPoint = [0.00, 0.00, 0.00, 0.00]
		trainingPoint[0] = 1.00		# Bias
		x1, x2 = np.random.uniform(-1, 1, 2)
		trainingPoint[1] = x1		# x1
		trainingPoint[2] = x2		# x2
		trainingPoint[3] = 0.0		# y output initialized to 0
		dataSet.append(trainingPoint)

	return dataSet

def plot(matrix, weights = None, title = "Prediction Matrix"):

	if len(matrix[0])==3: # if 1D inputs, excluding bias and ys
		fig,ax = plt.subplots()
		ax.set_title(title)
		ax.set_xlabel("i1")
		ax.set_ylabel("Classifications")

		if weights!=None:
			y_min=-1.1
			y_max=1.1
			x_min=-1.1
			x_max=1.1
			y_res=0.001
			x_res=0.001
			ys=np.arange(y_min,y_max,y_res)
			xs=np.arange(x_min,x_max,x_res)
			zs=[]
			for cur_y in np.arange(y_min,y_max,y_res):
				for cur_x in np.arange(x_min,x_max,x_res):
					zs.append(predict([1.0,cur_x],weights))
			xs,ys=np.meshgrid(xs,ys)
			zs=np.array(zs)
			zs = zs.reshape(xs.shape)
			cp=plt.contourf(xs,ys,zs,levels=[-1,-0.0001,0,1],colors=('b','r'),alpha=0.1)

		c1_data=[[],[]]
		c0_data=[[],[]]

		for i in range(len(matrix)):
			cur_i1 = matrix[i][1]
			cur_y  = matrix[i][-1]

			if cur_y==1:
				c1_data[0].append(cur_i1)
				c1_data[1].append(1.0)
			else:
				c0_data[0].append(cur_i1)
				c0_data[1].append(0.0)

		plt.xticks(np.arange(x_min,x_max,0.1))
		plt.yticks(np.arange(y_min,y_max,0.1))
		plt.xlim(-1.05,1.05)
		plt.ylim(-1.05,1.05)

		c0s = plt.scatter(c0_data[0],c0_data[1],s=40.0,c='r',label='Class -1')
		c1s = plt.scatter(c1_data[0],c1_data[1],s=40.0,c='b',label='Class 1')

		plt.legend(fontsize=10,loc=1)
		plt.show()
		return

	if len(matrix[0])==4: # if 2D inputs, excluding bias and ys
		fig,ax = plt.subplots()
		ax.set_title(title)
		ax.set_xlabel("i1")
		ax.set_ylabel("i2")

		if weights!=None:
			map_min=-1.1
			map_max=1.1
			y_res=0.001
			x_res=0.001
			ys=np.arange(map_min,map_max,y_res)
			xs=np.arange(map_min,map_max,x_res)
			zs=[]
			for cur_y in np.arange(map_min,map_max,y_res):
				for cur_x in np.arange(map_min,map_max,x_res):
					zs.append(predict([1.0,cur_x,cur_y],weights))
			xs,ys=np.meshgrid(xs,ys)
			zs=np.array(zs)
			zs = zs.reshape(xs.shape)
			cp=plt.contourf(xs,ys,zs,levels=[-1,-0.0001,0,1],colors=('b','r'),alpha=0.1)

		c1_data=[[],[]]
		c0_data=[[],[]]
		for i in range(len(matrix)):
			cur_i1 = matrix[i][1]
			cur_i2 = matrix[i][2]
			cur_y  = matrix[i][-1]
			if cur_y==1:
				c1_data[0].append(cur_i1)
				c1_data[1].append(cur_i2)
			else:
				c0_data[0].append(cur_i1)
				c0_data[1].append(cur_i2)

		plt.xticks(np.arange(-1.1,1.1,0.1))
		plt.yticks(np.arange(-1.1,1.1,0.1))
		plt.xlim(-1.05,1.05)
		plt.ylim(-1.05,1.05)

		c0s = plt.scatter(c0_data[0],c0_data[1],s=40.0,c='r',label='Class -1')
		c1s = plt.scatter(c1_data[0],c1_data[1],s=40.0,c='b',label='Class 1')

		plt.legend(fontsize=10,loc=1)
		plt.show()
		return

	print("Matrix dimensions not covered.")

def predict(inputs, weights):
	threshold = 0.0
	total_activation = 0.0

	for input,weight in zip(inputs,weights):
		total_activation += input*weight
	return 1.0 if total_activation >= threshold else 0.0

#   Calculate prediction accuracy, providad inputs
#   and associated weights...
def accuracy(matrix, weights):
	num_correct = 0.0
	preds = []

	for i in range(len(matrix)):
		# get predicted classification
		pred = predict(matrix[i][:-1], weights)
		preds.append(pred)
		# check if prediction is accurate
		if pred == matrix[i][-1]: num_correct += 1.0

	print("Predictions:", preds)
	# return overall prediction accuracy
	return num_correct/float(len(matrix))

#   Train the perceptron on the data found in the Matrix
#   trained weights are returned at the end of the function
def train_weights(matrix, weights, nb_epoch = 10, l_rate = 1.0,
			do_plot = False, stop_early = True, verbose = True):
	# iterate for the number of epochs requested
	for epoch in range(nb_epoch):
		# calculate the current accuracy
		cur_acc = accuracy(matrix, weights)
		# print out information
		print("\nEpoch %d \nWeights: " %epoch, weights)
		print("Accuracy: ", cur_acc)

		# check if we are finished training
		if cur_acc == 1.0 and stop_early: break

		# check if we should plot the current results
		if do_plot: plot(matrix, weights, title = "Epoch %d" %epoch)

		for i in range(len(matrix)):
			# calculate Prediction
			prediction = predict(matrix[i][:-1], weights)
			# calculate error
			error = matrix[i][-1] - prediction

			#if error != 0:
			if verbose:
				print("Training on data at index %d..." %i)

			for j in range(len(weights)):
				if verbose:
					sys.stdout.write("\tWeight[%d]: %0.5f --> " %(j,weights[j]))

				weights[j] = weights[j] + (l_rate * error * matrix[i][j])

				if verbose:
					sys.stdout.write("%0.5f\n" %weights[j])

	# plot out the final results
	plot(matrix, weights, title = "Final Epoch")
	return epoch

def main():
	#	This is in case the experiment needs to be run multiple times
	loops = range(1)
	iterations = len(loops)
	sum = 0

	for test in loops:
		#	Generating data
		data = genData(100)
		classifLine = genLine()
		classifyTrainingPoint(data, classifLine)
		weights	= [0.00, 0.00, 0.00]

		#	Executing algorithm
		iteration = train_weights(data, weights = weights, nb_epoch = 10000, l_rate = 1.0,
					do_plot = False, stop_early = True, verbose = False)

		sum += iteration

	avg = sum / iterations

	print("It took %d iterations on average" %avg)

if __name__ == '__main__':
	main()
