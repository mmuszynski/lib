#! /usr/bin/env python3
# H+
#	Title   : view_tester.py
#	Author  : Matt Muszynski
#	Date    : 06/30/17
#	Synopsis: 
# 
#
#	$Date$
#	$Source$
#  @(#) $Revision$
#	$Locker$
#
#	Revisions:
#
# H-
# U+
#	Usage   :
#	Example	:
#	Output  :
# U-
# D+
#
# D-
###############################################################################

from numpy import deg2rad, outer, linspace, sin, cos, ones, vstack, size
from numpy import zeros, arange, hstack
from numpy import meshgrid, asarray

###############################################################################
#
#	colVec() is a function to create a numpy column vector. It's a bit
#		friendlier than doing it manually since you have to explictly
#		tell numpy dimension in order to get the right shape.
#
###############################################################################
def colVec(listLike):
	listLike = asarray(listLike)
	return listLike[None].T


###############################################################################
#
#	tilde() is a function
#
###############################################################################
def tilde(threevec):
        import numpy as np
        x1, x2, x3 = threevec
        return np.array([
                [  0,-x3, x2],
                [ x3,  0,-x1],
                [-x2, x1,  0]
                ])

###############################################################################
#
#	sphereSample() is a function
#
###############################################################################

def sphereSample(thetaBins, phiBins):
	lons = deg2rad(linspace(-180, 180, thetaBins))
	lats = deg2rad(linspace(0, 180, phiBins)[::-1])
	lats, lons = meshgrid(lats,lons)
	lons = lons.reshape(1,-1)[0]
	lats = lats.reshape(1,-1)[0]

	e1 = cos(lons)*sin(lats)
	e2 = sin(lons)*sin(lats)
	e3 = cos(lats)

	e = vstack([e1,e2,e3])

	return e

###############################################################################
#	block_diag() is used to create a block diagonal matrix given a set of
#	smaller matrices. It takes a numpy array in the form:
#
#			[  0   1   2   3   4   5]
#			[  6   7   8   9  10  11]
#	 		[ 12  13  14  15  16  17]
#
#	and turns it into the form:
#
#			[  0   1   2   0   0   0]
#			[  6   7   8   0   0   0]
#	 		[ 12  13  14   0   0   0]
#	 		[  0   0   0   3   4   5]
#	 		[  0   0   0   9  10  11]
#	 		[  0   0   0  15  16  17]
#
#	Inputs:
#		in_mat: an nxm numpy array consisting of m/n nxn matrices.
#
#	Outputs:
#		out_mat: an mxm array with the same nxn matrices that were passed in
#		but now buffered with zeros to make it block diagonal.
#
#	Notes:
#		This is a helper function to manipulate very large sets of data. I
#		originally wrote it so I can do many coe2rv calculations all at once.
#
###############################################################################

def block_diag(in_mat):

	from numpy import arange, zeros
	#nxm array --> mxm array (n<m)
	n = len(in_mat)
	m = len(in_mat[0])

	if m%n:
		print("Error: input array must be nxm where n divides evenly into m.")
		return

	nm_helper = arange(n*m).reshape(n,m)
	mm_helper = arange(m**2)
	m_helper = arange(m)
	out_mat = zeros(m**2)
	
	for i in range(0,n):
		for j in range(n):
			out_mat[mm_helper%((m+1)*n) == m*i+j]  = \
				in_mat[i][m_helper%n ==j]
	out_mat = out_mat.reshape(m,m)

	return out_mat

################################################################################
#	Rotation matrices
#
# 	Reference: lecture 6 ASEN 5050 CU Boulder, Fall 2016, Slide 37
#
#	Angles all in radians!
#
###############################################################################

# rx =  np.matrix( \
# [ \
# [1.,  0.,         0.        ], \
# [0.,  np.cos(theta), np.sin(theta)], \
# [0., -np.sin(theta), np.cos(theta)]  \
# ] \
# )	

def rx (theta):

	try:
		length = len(theta)
		zero = zeros(length)
		one = ones(length)
	except:
		length = 1
		zero = 0
		one = 1

	rx00 =  one
	rx01 =  zero
	rx02 =  zero
	rx10 =  zero
	rx11 =  cos(theta)
	rx12 =  sin(theta)
	rx20 =  zero
	rx21 = -sin(theta)
	rx22 =  cos(theta)


	if length == 1:
		rx = hstack(
			[rx00,rx01,rx02,rx10,rx11,rx12,rx20,rx21,rx22]
			).reshape(3,3)
	else:
		rx = vstack(
			[rx00,rx01,rx02,rx10,rx11,rx12,rx20,rx21,rx22]
			).T.reshape(length,3,3)
	return rx


def ry(theta):

	try:
		length = len(theta)
		zero = zeros(length)
		one = ones(length)
	except:
		length = 1
		zero = 0
		one = 1

	ry00 =  cos(theta)
	ry01 =  zero
	ry02 = -sin(theta)
	ry10 =  zero
	ry11 =  one
	ry12 =  zero
	ry20 =  sin(theta)
	ry21 =  zero
	ry22 =  cos(theta)

	if length == 1:
		ry = hstack(
			[ry00,ry01,ry02,ry10,ry11,ry12,ry20,ry21,ry22]
			).reshape(3,3)
	else:
		ry = vstack(
			[ry00,ry01,ry02,ry10,ry11,ry12,ry20,ry21,ry22]
			).T.reshape(length,3,3)
	return ry

def rz (theta):

	try:
		length = len(theta)
		zero = zeros(length)
		one = ones(length)
	except:
		length = 1
		zero = 0
		one = 1

	rz00 =  cos(theta)
	rz01 =  sin(theta)
	rz02 =  zero
	rz10 = -sin(theta)
	rz11 =  cos(theta)
	rz12 =  zero
	rz20 =  zero
	rz21 =  zero
	rz22 =  one

	if length == 1:
		rz = hstack(
			[rz00,rz01,rz02,rz10,rz11,rz12,rz20,rz21,rz22]
			).reshape(3,3)
	else:
		rz = vstack(
			[rz00,rz01,rz02,rz10,rz11,rz12,rz20,rz21,rz22]
			).T.reshape(length,3,3)
	return rz

def r1 (theta):
	r1 = rx(theta)
	return r1

def r2 (theta):
	r2 = ry(theta)
	return r2

def r3 (theta):
	r3 = rz(theta)
	return r3


###############################################################################
#
# interpolateLambdaDependent() 
#
#	Inputs:
#		
#	Outputs:
#
#	Notes: Please forgive me for my crappy variable names in this method.
#		at least it's short and relatively simple...
#
###############################################################################

def interpolateLambdaDependent(ex,lambda_set):
	from numpy import array
	lam = ex['lambda']
	data = ex['throughput']

	int_ex = []
	lambda_set_ex = []
	for i in range(0,len(lambda_set)):
		#if this item in lambda_set is in the lambda array passed
		#by the user, just grab its data value and use it.
		if min(abs(lambda_set[i] - lam)) < 1e-8:
			for j in range(0,len(lam)):
				if lam[j] == lambda_set[i]:
					data_ex	= data[j]
		#if this item in lambda_set is less than the minimum of the
		#lambda array passed by the user then this curve has no
		#throughput at this wavelength. Set data to zero.					
		elif lambda_set[i] < min(lam):
			data_ex = 0
		#if this item in lambda_set is greater than the maximum of the
		#lambda array passed by the user then this curve has no
		#throughput at this wavelength. Set data to zero.					
		elif lambda_set[i] > max(lam):
			data_ex = 0
		else:
		#this is the meat of this method. If this item in lambda_set is
		#not already represented by a point in the 'lambda' array passed
		#by the user, then take the point just above and just below it
		#and do a linear interpolation between them to find a representation
		#of throughput at the given lambda.
			for j in range(0,len(lam)):
				if lam[j] < lambda_set[i]:
					lower_lam = lam[j]
					lower_data = data[j]
					upper_lam = lam[j+1]
					upper_data = data[j+1]	
					m = (upper_data-lower_data)/(upper_lam-lower_lam)		
			data_ex = lower_data + m*(lambda_set[i]-lower_lam)
		lambda_set_ex.insert(len(lambda_set_ex),lambda_set[i])
		int_ex.insert(len(int_ex),data_ex)
	return {
	'lambda': array(lambda_set_ex),
	'throughput': array(int_ex)
}

###########################################################################
#
# rasterize() floors the pixel and line coordinates and the uses pandas
#		to sum all intensity that falls in the same bin.
#
###########################################################################
def rasterize(pixelResolution,lineResolution,pixelCoord,lineCoord,intensity, **kwargs):
	"""!
	@param pixelResolution: number of pixels in the width dimension of the
		detector array
	@param lineResolution: number of pixels in the height dimension of the
		detector array
	@param pixelCoord: x (pixel) coordinate of every point source in scene
	@param lineCoord: y (line) coordinate of every point source in scene
	@param intensity: incident intensity of every point source in scene

	@return detectorArray: array with summed intenisty for every pixel in
		the detector array
	"""
	from numpy import floor, zeros, array, arange, append 
	from numpy import concatenate, logical_and
	from pandas import DataFrame

	try:
		avg = kwargs['avg']
	except:
		avg = 0
		
	#adding PSF introduces some values that are not on the detector. Remove them here

	positiveCoords = logical_and(pixelCoord > 0, lineCoord > 0)
	pixelCoord = pixelCoord[positiveCoords]
	lineCoord = lineCoord[positiveCoords]
	intensity = intensity[positiveCoords]

	notTooBig = logical_and(pixelCoord < pixelResolution, lineCoord < lineResolution)
	pixelCoord = pixelCoord[notTooBig]
	lineCoord = lineCoord[notTooBig]
	intensity = intensity[notTooBig]

	intPixCoord = floor(pixelCoord).astype(int)
	intLineCoord = floor(lineCoord).astype(int)

	detectorPosition = (lineResolution*intLineCoord + intPixCoord)
	detectorPosition = append(detectorPosition,arange(pixelResolution*lineResolution))
	intensity = append(intensity,zeros(pixelResolution*lineResolution))

	data = concatenate([detectorPosition,intensity])
	data = data.reshape(2,int(len(data)/2)).T
	df = DataFrame(data,columns = ["Position","Intensity"])

	if avg == 1:
		detectorArray = df.groupby("Position").mean().values.T[0]
	else:
		detectorArray = df.groupby("Position").sum().values.T[0]

	return detectorArray
