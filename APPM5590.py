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

from numpy import sqrt

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



