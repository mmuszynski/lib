#! /usr/bin/env python3
###############################################################################
#
#	Title   : APPM5590.py
#	Author  : Matt Muszynski
#	Date    : 01/29/18
#	Synopsis: Functions for APPM 5990. I could probably do all this through
#		scipy and pandas, but I feel more comfortable learning it from the
#		inside out.
#
###############################################################################

from numpy import sqrt, genfromtxt, hstack, ones, hstack, array, diag
from numpy import setdiff1d, arange, unique, empty, delete, vstack
from numpy import savetxt, log, set_printoptions
from os import remove
from scipy.stats import probplot
from numpy.linalg import inv
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MLR:
	
	def __init__(self):
		self.X = -1
		self.Y = -1
		self.betaHat = -1
		self.partialRegressionSwitches = []
		self.partialRegressions = []
		self.fTest = []
		self.fDOF = []

	def parseData(self,ignoreVars,rowLabelCol,yCol):
		'''!
		Note: rowLabelCol is a switch to tell the method whether or not
		to use the first column as labels for the rows. Sometimes it's
		used as part of an index.
		'''


		data = genfromtxt(self.dataFile)
		self.rawData = data
		tossTheseColumns = hstack([array([rowLabelCol,yCol]),ignoreVars])
		allColumns = arange(data.shape[1])
		allRows = arange(data.shape[0])
		keepTheseColumns = setdiff1d(allColumns,tossTheseColumns)
		headerRow = 0
		tossTheseRows = array([headerRow])
		keepTheseRows = setdiff1d(allRows,tossTheseRows)

		headers = genfromtxt(self.dataFile,dtype=str)[0,keepTheseColumns]
		self.yName = genfromtxt(self.dataFile,dtype=str)[0,yCol]
		self.dataNames = headers
		self.Y = data[keepTheseRows,yCol,None]
		self.X = hstack([ones(len(keepTheseRows)).reshape(-1,1),
					data[keepTheseRows][:,keepTheseColumns]])
		if rowLabelCol != -1:
			self.rowLabel = data[:,rowLabelCol,None][1:]
			self.rowLabelName = genfromtxt(self.dataFile,dtype=str)[0,0]
		else:
			self.rowLabel = -1

	def calculateBetaHat(self):
		'''!
		RABE Chapter 3 eqn A.3. Page 82
		'''
		Y = self.Y
		X = self.X

		self.betaHat = inv(X.T.dot(X)).dot(X.T).dot(Y)

	def calculateYhat(self):
		'''!
		RABE Chapter 3 eqn A.4. Page 82
		'''
		X = self.X
		betaHat = self.betaHat
		self.Yhat = X.dot(betaHat)

	def calculateHMatrix(self):
		'''!
		RABE Chapter 3 eqn A.5. Page 83
		'''
		X = self.X
		self.H = X.dot(inv(X.T.dot(X))).dot(X.T)


	def calculateResiduals(self):
		'''!
		RABE Chapter 3 eqn A.6. Page 83
		'''
		self.e = self.Y - self.Yhat

	def calculateCMatrix(self):
		'''!
		RABE Chapter 3 eqn A.7. Page 83
		'''
		X = self.X
		self.C = inv(X.T.dot(X))

	def calculateSigmaHat(self):
		e = self.e
		n = len(self.Y)
		p = len(self.betaHat) - 1
		self.sigmaHat = sqrt((e.T.dot(e)/(n-p-1))[0,0])

	def calculateSeBeta(self):
		sigmaHat = self.sigmaHat
		C = self.C
		self.seBetaJ = sigmaHat*sqrt(diag(C).reshape(-1,1))

	def calculateSSE(self):
		'''!
		RABE 3.26
		'''
		e = self.e
		self.SSE = e.T.dot(e)[0,0]

	def calculateSST(self):
		'''!
		RABE 2.44
		'''
		from numpy import mean
		Y = self.Y
		self.SST = (Y - mean(Y)).T.dot(Y - mean(Y))[0,0]

	def calculateSSR(self):
		'''!
		RABE 2.44
		'''
		from numpy import mean
		Yhat = self.Yhat
		Y = self.Y
		self.SSR = (Yhat - mean(Y)).T.dot(Yhat - mean(Y))[0,0]

	def calculateRSq(self):
		'''!
		RABE 3.17
		'''
		self.rSq = self.SSR/self.SST

	def calculateTTest(self):
		self.tTest = self.betaHat/self.seBetaJ

	def calculateStudentizedResiduals(self):
		e = self.e
		sigmaHat = self.sigmaHat
		h = diag(self.H).reshape(-1,1)
		r = e/(sigmaHat*sqrt(1-h))
		self.r = r

	def runFTest(self,partialRegression,**kwargs):
		'''!
		RAE eqn 3.28
		'''
		try:
			saveResult = kwargs['saveResult']
		except:
			saveResult = 1

		SSERM = partialRegression.SSE
		SSEFM = self.SSE
		#this is correct. Full model has p+1 parameters, and
		#reduced model has k parameters. Therefore p == k if
		#we reduce the model by one parameter
		p = self.X.shape[1]-1
		k = partialRegression.X.shape[1]
		n = self.Y.shape[0]
		F = (SSERM - SSEFM)/(SSEFM)*(n-p-1)/(p+1-k)
		dof = (p+1-k,n-p-k)
		if saveResult:
			self.fTest.append(F)
			self.fDOF.append(dof)
		else:
			return {
				'SSERM': SSERM,
				'SSEFM': SSEFM,
				'p': p,
				'k': k,
				'n': n,
				'F': F,
				'dof': 	dof
				}

	def separateCatVars(self,catVars,**kwargs):
		if catVars == -1: return
		try:
			keepVals = kwargs['keepVals']
		except:
			keepVals = 0
		n = self.X.shape[0]
		m = self.X.shape[1]
		newX = ones((n,1))
		newDataNames = []

		###############################################################
		#
		# note one why it's m-1. e.g.: If we have 3 categorical values,
		#  we only need 2 variables to track them all. If we know
		#  the values of v1 and v2, the value of v3 is determined.
		#  reducing the number of variables by 1 allows us to keep one
		#  more degree of freedom. We don't care which one we keep, so
		#  the code just choses to throw out the last one by iterating
		#  though one i too few to get them all.
		#
		###############################################################

		for i in range(1,m):

			if i in catVars:
				oldCol = self.X[:,i,None]
				uniqVals = unique(oldCol)
				for j in uniqVals[0:-1]:
					#need to call newCol explicitly as an array
					#so we don't change things passing by reference
					newCol = array(oldCol)
					yesInd = newCol == j
					noInd = newCol != j
					newCol[noInd] = 0
					#normalize values if we decided not to keep them
					#this is the nominal behavior
					if keepVals == 0:
						newCol[yesInd] = 1

					
					newX = hstack([newX,newCol])
					newDataNames = hstack([
						newDataNames,
						self.dataNames[i-1][0] + str(int(j))
						])
			else:
				newX = hstack([newX,self.X[:,i,None]])
				newDataNames = hstack([newDataNames,self.dataNames[i-1]])

		self.X = newX
		self.dataNames = newDataNames


	def createInteractionTerms(self,intVars,**kwargs):
		if intVars == -1: return
		try:
			keepVals = kwargs['keepVals']
		except:
			keepVals = 0

		for interaction in intVars:
			self.X = hstack([
				self.X,
				self.X[:,interaction[0],None]*\
				self.X[:,interaction[1],None] 
				])
			self.dataNames = hstack([
				self.dataNames,
				self.dataNames[interaction[0]-1] + '*' + \
				self.dataNames[interaction[1]-1]
				])

	def removeBadObservations(self,badObs):
		if badObs == -1: return
		badObs = array(badObs)
		self.X = delete(self.X,badObs,0)
		self.Y = delete(self.Y,badObs,0)

	def transformY(self,yTransform):
		self.yTransform = yTransform
		if yTransform == 'ln':
			self.Y = log(self.Y)
			self.yName = 'ln' + self.yName

	def regress(self,**kwargs):
		try:
			ignoreVars = kwargs['ignoreVars']
		except:
			ignoreVars = array([-1])
		try:
			rowLabelCol = kwargs['rowLabelCol']
		except:
			rowLabelCol=-1
		try:
			yCol = kwargs['yCol']
		except:
			yCol = 0
		try:
			catVars = kwargs['catVars']
		except:
			catVars = -1
		try:
			intVars = kwargs['intVars']
		except:
			intVars = -1
		try:
			badObs = kwargs['badObs']
		except:
			badObs = -1
		try:
			yTransform = kwargs['yTransform']
		except:
			yTransform = -1

		self.parseData(ignoreVars,rowLabelCol,yCol)
		self.removeBadObservations(badObs)
		self.separateCatVars(catVars)
		self.createInteractionTerms(intVars)
		self.transformY(yTransform)
		self.calculateBetaHat()
		self.calculateYhat()
		self.calculateHMatrix()
		self.calculateResiduals()
		self.calculateCMatrix()
		self.calculateSigmaHat()
		self.calculateSeBeta()
		self.calculateSST()
		self.calculateSSR()
		self.calculateSSE()
		self.calculateRSq()
		self.calculateTTest()
		self.calculateStudentizedResiduals()

	def partialRegress(self,ignoreVars):

		try: 
			rowLabelCheck = (self.rowLabel == -1).all()
		except:
			rowLabelCheck = self.rowLabel == -1

		if rowLabelCheck:
			header = hstack([self.yName,self.dataNames])
			data = hstack([self.Y,self.X[:,1:]]) 
			rowLabelCol = -1
			yCol = 0
		else:
			header = hstack([self.rowLabelName,self.yName,self.dataNames])
			data = hstack([self.rowLabel,self.Y,self.X[:,1:]]) 
			rowLabelCol = 0
			yCol = 1
		allTogetherNow = vstack([header,data])
		savetxt('/tmp/tmp.txt',allTogetherNow,fmt='%s',delimiter="\t")
		#construct a temporary data file to leverage the way the
		#regress method collects data.

		partialRegression = MLR()
		partialRegression.dataFile = '/tmp/tmp.txt'
		partialRegression.regress(
			ignoreVars=ignoreVars,
			rowLabelCol=rowLabelCol,
			yCol=yCol
			)
		remove('/tmp/tmp.txt')
		self.partialRegressionSwitches.append(ignoreVars)
		self.partialRegressions.append(partialRegression)
		self.runFTest(partialRegression)

	def scatterPlot(self,regression):
		if regression.X.shape[1] != 2:
			print('Error. Scatter only available for 2D data.')
			return
		plt.figure()
		plt.plot(regression.X[:,1],regression.Y,'.',label='Raw Data')
		plt.plot(regression.X[:,1],regression.Yhat,'r',label='Fitted Data')
		plt.ylabel(regression.yName)
		plt.xlabel(regression.dataNames[0])
		plt.title(regression.yName + \
			' versus ' + regression.dataNames[0])
		plt.legend()


	def scatterAllPartials(self):
		for i in range(0,len(self.partialRegressions)):
			plt.figure()
			self.scatterPlot(self.partialRegressions[i])

	def scatter3D(self):
		if self.X.shape[1] != 3:
			print('Error. 3D scatter only available for 3D data.')
			return
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.scatter(
			self.X[:,1], 
			self.X[:,2], 
			self.Y.T[0], label='Raw Data')
		ax.scatter(
			self.X[:,1], 
			self.X[:,2], 
			self.Yhat.T[0], 
			color='r',label='Fitted Data')
		ax.set_title(self.yName + ' versus ' + \
			self.dataNames[0] + ' and ' + self.dataNames[1])
		ax.set_xlabel(self.dataNames[0])
		ax.set_ylabel(self.dataNames[1])
		ax.set_zlabel(self.yName)
		ax.legend()

	def plotMatrix(self):
		fullData = hstack([self.Y,self.X[:,1:]])
		dim = fullData.shape[1]
		fig, axes = plt.subplots(
			dim, dim, sharex='col', sharey='row')
		names = hstack([self.yName,self.dataNames])
		for i in range(0,dim):
			for j in range(0,dim):
				if i == j:
					text = names[i]
					axes[i,j].text(0.5, 0.5,text,
						horizontalalignment='center',
						verticalalignment='center',
						transform=axes[i,j].transAxes)
				elif i > j:
					text = "{:.3f}".format(
						cor(fullData[:,i],fullData[:,j])
						)
					
					axes[i,j].text(0.5, 0.5,text,
						horizontalalignment='center',
						verticalalignment='center',
						transform=axes[i,j].transAxes)

				else:
					axes[i,j].plot(fullData[:,j],fullData[:,i],'.')


	def plotResidualsVPredictors(self,dims):
		X = self.X[:,1:]
		r = self.r
		fig, axes = plt.subplots(
			dims[0], dims[1], sharex=False, sharey=True)
		plt.suptitle('Standardized (Studentized) Residuals versus Predictor Variables')
		plt.subplots_adjust(hspace=0.5)
		if dims[1] == 1:
			for i in range(0,dims[0]):
					axes[i].plot(X[:,i],r,'.')
					axes[i].set_xlabel(self.dataNames[i])
					axes[i].set_ylabel('Residuals')
		else:		
			for i in range(0,dims[0]):
				for j in range(0,dims[1]):
					axes[i,j].plot(X[:,i*dims[1]+j],r,'.')
					axes[i,j].set_xlabel(self.dataNames[i*dims[1]+j])
					axes[i,j].set_ylabel('Residuals')

	def plotResidualsVFitted(self):
		Yhat = self.Yhat
		r = self.r.T[0]
		plt.figure()
		plt.plot(Yhat,r,'.')
		plt.title('Standardized (Studentized) Residuals versus Fitted Values')
		plt.xlabel('Fitted Values')
		plt.ylabel('Residuals')

	def plotStandardizedResiduals(self):
		r = self.r.T[0]
		plt.figure()
		plt.plot(r,'.')
		plt.title('Standardized (Studentized) Residuals versus Observation Index')
		plt.xlabel('Index')
		plt.ylabel('Residual')

	def plotNormalProb(self):
		r = self.r.T[0]
		plt.figure()
		probplot(r.T,plot=plt)

	def predict(self,x0):
		yMuHat = x0.T.dot(self.betaHat)[0,0]
		seYHat = self.sigmaHat*sqrt(
				1 + x0.T.dot(inv(self.X.T.dot(self.X)).dot(x0)
				)
			)[0,0]
		seMuHat = self.sigmaHat*sqrt(
				x0.T.dot(inv(self.X.T.dot(self.X)).dot(x0)
				)
			)[0,0]
		print("yHat/muHat: " + str(yMuHat))
		print("se(yHat): " + str(yMuHat))
		print("se(muHat): " + str(seYHat))

		return (yMuHat, seYHat, seMuHat)

	def latexTable(self,**kwargs):
		print("\\begin{table}[H]")
		print("\centering")
		print("\\begin{tabular}{|c|c|c|c|}")
		print("\hline")
		print("Predictor & $\hat{\\beta}_i$ & se($\hat{\\beta}_i$) & t-test \\\\")
		print("\\hline")
		names = hstack(["Constant",self.dataNames])
		for i in range(0,len(self.betaHat)):
			print(
				names[i] + " & " + \
				str("%.3E" % self.betaHat[i][0]) + " & " + \
				str("%.3E" % self.seBetaJ[i][0]) + \
				" & " + \
				str("%.3E" % self.tTest[i][0]) + " \\\\"
				)
		print("\hline")
		print(
			"$n=$" + str(len(self.Y)) + " & " + \
			"$R^2=$" + str("%.3f" % self.rSq) + " & " + \
			"$\\hat{\\sigma}$=" + str("%.3f" % self.sigmaHat) + " & " + \
			"d.f.=" + str(
				len(self.Y)-len(self.betaHat)
				) + " & "
			)
		print("\hline")
		print("\end{tabular}")
		print("\caption{My caption}")
		print("\label{my-label}")
		print("\end{table}")

def bar(X):
	'''!
	It's a mean, duh. Just doing this for pedagogical reasons.
	RABE 2.1
	'''
	X = X.astype(float)
	return sum(X)/(len(X))

def cov(Y,X):
	'''!
	RABE 2.2
	'''
	X = X.astype(float)
	Y = Y.astype(float)

	Ybar = bar(Y)
	Xbar = bar(X)
	n = len(Y)
	return sum((Y-Ybar)*(X-Xbar))/(n-1)

def z(Y):
	'''!
	RABE 2.3
	'''
	Y = Y.astype(float)
	Ybar = bar(Y)
	sigmaY = std(Y)
	return (Y - Ybar)/sigmaY

def std(Y):
	'''!
	RABE 2.4
	'''
	Y = Y.astype(float)
	Ybar = bar(Y)
	n = len(Y)
	return sqrt(sum((Y-Ybar)**2)/(n-1))

def cor(Y,X):
	'''!
	RABE 2.6. Equivalent to RABE 2.5 and 2.7.
	'''
	Y = Y.astype(float)
	X = X.astype(float)
	sigmaY = std(Y)
	sigmaX = std(X)
	covYX = cov(Y,X)
	return covYX/(sigmaY*sigmaX)

def simpleLR(Y,X,**kwargs):
	'''!
	A lot of stuff in here
	beta1Hat is from RABE 2.14
	beta0Hat is from RABE 2.15
	Yhat is from RABE 2.16
	e is from RABE 2.18
	sigmaHatSq is from RABE 2.23
	seBeta0Hat is from RABE 2.24
	seBeta1Hat is from RABE 2.25
	'''
	Y = Y.astype(float)
	X = X.astype(float)

	try: 
		beta00 = kwargs['beta00']
	except:
		beta00 = 0

	try: 
		beta10 = kwargs['beta10']
	except:
		beta10 = 0

	try: 
		criticalT = kwargs['criticalT']
	except:
		criticalT = 0
	
	try: 
		throughOrigin = kwargs['throughOrigin']
	except:
		throughOrigin = 0

	Ybar = bar(Y)
	Xbar = bar(X)

	if throughOrigin:
		#this is not implemented. It probably never will be, but
		#here's a placeholder for it if I decide to someday
		return -1
	else:
		beta1Hat = sum((Y-Ybar)*(X-Xbar))/sum((X-Xbar)**2)
		beta0Hat = Ybar - beta1Hat*Xbar

		Yhat = beta0Hat + beta1Hat*X
		e = Y - Yhat
		SST = sum((Y - Ybar)**2)
		SSR = sum((Yhat - Ybar)**2)
		SSE = sum(e**2)
		n = float(len(Y))
		sumSqXDiff = sum((X-Xbar)**2)
		sigmaHatSq = SSE/(n - 2)
		sigmaHat = sqrt(sigmaHatSq)

		#why are these two different?
		seBeta0Hat = sigmaHat*sqrt(1/n + Xbar**2/sum((X-Xbar)**2))
		
		seBeta1Hat = sigmaHat/sqrt(sumSqXDiff)
		t1 = (beta1Hat - beta10)/seBeta1Hat
		t0 = (beta0Hat - beta00)/seBeta0Hat
		beta0HatPM = criticalT*seBeta0Hat
		beta1HatPM = criticalT*seBeta1Hat
		Rsq = SSR/SST

	#I'll probably regret returning this as a dict some day. But today
	#I don't know enough about what I'm doing to make a better decision.
	return {
		'beta0Hat': beta0Hat,
		'beta1Hat': beta1Hat,
		'seBeta0Hat': seBeta0Hat,
		'seBeta1Hat': seBeta1Hat,	
		'sigmaHat': sigmaHat,	
		'Yhat': Yhat,
		'n': n,
		'e': e,
		'SSE': SSE,
		'SST': SST,
		'SSR': SSR,
		't1': t1,
		't0': t0,
		'beta0HatPM': beta0HatPM,
		'beta1HatPM': beta1HatPM,
		'Xbar': Xbar,
		'Ybar': Ybar,
		'X': X,
		'Y': Y
	}


def simpleLREstimate(simpleLROutput,x0,criticalT):
	'''!
	RABE p37-8
	'''

	beta0Hat = simpleLROutput['beta0Hat']
	beta1Hat = simpleLROutput['beta1Hat']
	sigmaHat = simpleLROutput['sigmaHat']
	Xbar = simpleLROutput['Xbar']
	n = simpleLROutput['n']
	X = simpleLROutput['X']

	yHat0 = beta0Hat + beta1Hat*x0
	muHat0 = beta0Hat + beta1Hat*x0
	seY0Hat = sigmaHat*sqrt(1 + n**-1 + (x0 - Xbar)**2/sum((X-Xbar)**2))
	seMu0Hat = sigmaHat*sqrt(n**-1 + (x0 - Xbar)**2/sum((X-Xbar)**2))

	yHat0PM = criticalT*seY0Hat
	muHat0PM = criticalT*seMu0Hat

	return {
		'yHat0': yHat0,
		'muHat0': muHat0,
		'seY0Hat': seY0Hat,
		'seMu0Hat': seMu0Hat,
		'yHat0PM': yHat0PM,
		'muHat0PM': muHat0PM
	}



