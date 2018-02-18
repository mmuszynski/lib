#! /usr/bin/env python3
# H+
#	Title   : rigidBodyKinematics.py
#	Author  : Matt Muszynski
#	Date    : 01/04/18
#	About	: rigidBodyKinematics.py is largely compiled from the course
#			  notes written by Dr. Hanspeter Schaub for his ASEN 5010 course
#			  (Spacecraft Attitude Dynamics and Control) as taught Spring 2017.
#
###############################################################################

# note to self. Good Test for DCMs is C^-1 = C^T
# note to self. Good Test for DCMs det(C) = 1 (is -1 if left-handed)

from util import tilde
from numpy import array, sin, cos, arctan2, arcsin, arccos, rad2deg, deg2rad
from numpy import identity, outer, trace
from numpy.linalg import norm
###############################################################################
#
#	CDot() is a simple funciton to return the time derivative of a DCM.
#
#	inputs:
#		omega: numpy 3-length array of elements the C-matrix's rotational
#			velocity
#		C: the DCM that is rotating
#
#	outputs:
#		CDot: time derivative of the DCM
#
###############################################################################

def CDot(omega,C):
	CDOT = -tilde(omega).dot(C)
	return CDot

###############################################################################
#
#	r1() gives a DCM describing a rotation about the 1st axis
#
#	inputs:
#		theta: angle in degrees as float
#
#	outputs:
#		C: DCM for the rotation as 3x3 numpy array
#
###############################################################################

def r1(theta):
	theta = deg2rad(theta)
	C = array([
		[1,           0,          0],
		[0,  cos(theta), sin(theta)],
		[0, -sin(theta), cos(theta)]
		])
	return C

###############################################################################
#
#	r2() gives a DCM describing a rotation about the 2nd axis
#
#	inputs:
#		theta: angle in degrees as float
#
#	outputs:
#		C: DCM for the rotation as 3x3 numpy array
#
###############################################################################

def r2(theta):
	theta = deg2rad(theta)
	C = array([
		[cos(theta), 0, -sin(theta)],
		[         0, 1,           0],
		[sin(theta), 0,  cos(theta)]
		])
	return C

###############################################################################
#
#	r3() gives a DCM describing a rotation about the 3rd axis
#
#	inputs:
#		theta: angle in degrees as float
#
#	outputs:
#		C: DCM for the rotation as 3x3 numpy array
#
###############################################################################

def r3(theta):
	theta = deg2rad(theta)
	C = array([
		[  cos(theta),  sin(theta), 0],
		[ -sin(theta),  cos(theta), 0],
		[           0,           0, 1]
		])
	return C

###############################################################################
#
#	e3212C() gives a DCM describing a 3-2-1 euler angle rotation
#
#	inputs:
#		angles: 3-length numpy array containing rotations about each axis
#			**must be in 3-2-1 order!**
#
#	outputs:
#		C: DCM for the rotation as 3x3 numpy array
#
###############################################################################


def e3212C(angles):
	alpha = angles[0]
	beta = angles[1]
	gamma = angles[2]
	C = r1(gamma).dot(r2(beta).dot(r3(alpha)))
	return C


###############################################################################
#
#	C2e321() gives a set of 3-2-1 euler angles describing the same rotation
#		as the DCM C
#
#	inputs:
#		C: DCM for the rotation as 3x3 numpy array
#
#	outputs:
#		angles: 3-length numpy array containing rotations about each axis
#			**these are in 3-2-1 order!**
#
#
###############################################################################


def C2e321(C):
	alpha = arctan2(C[0,1],C[0,0])
	beta = -arcsin(C[0,2])
	gamma = arctan2(C[1,2],C[2,2])
	angles = array([alpha,beta,gamma])
	angles = rad2deg(angles)
	return angles

###############################################################################
#
#	e3132C() gives a DCM describing a 3-1-3 euler angle rotation
#
#	inputs:
#		angles: 3-length numpy array containing rotations about each axis
#			**must be in 3-1-3 order!**
#
#	outputs:
#		C: DCM for the rotation as 3x3 numpy array
#
###############################################################################


def e3132C(angles):
	alpha = deg2rad(angles[0])
	beta = deg2rad(angles[1])
	gamma = deg2rad(angles[2])
	C = r3(gamma).dot(r1(beta).dot(r3(alpha)))
	return C


###############################################################################
#
#	C2e313() gives a set of 3-1-3 euler angles describing the same rotation
#		as the DCM C
#
#	inputs:
#		C: DCM for the rotation as 3x3 numpy array
#
#	outputs:
#		angles: 3-length numpy array containing rotations about each axis
#			**these are in 3-1-3 order!**
#
#
###############################################################################


def C2e313(C):
	alpha = arctan2(C[2,0],-C[2,1])
	beta = arccos(C[2,2])
	gamma = arctan2(C[1,2],C[1,2])
	angles = array([alpha,beta,gamma])
	angles = rad2deg(angles)
	return angles

###############################################################################
#
#	e321dot2omega() 
#
#	inputs:
#
#	outputs:
#
#
###############################################################################

def e321dot2omega(e321dot, angles):
	psi = deg2rad(angles[0])
	theta = deg2rad(angles[1])
	phi = deg2rad(angles[2])

	B = array([
		[         -sin(theta),         0, 1],
		[ sin(phi)*cos(theta),  cos(phi), 0],
		[ cos(phi)*cos(theta), -sin(phi), 0]
		])
	omega = B.dot(e321dot)
	return omega

###############################################################################
#
#	omega2e321dot()
#
#	inputs:
#
#	outputs:
#
#
###############################################################################


def omega2e321dot(omega, angles):
	psi = deg2rad(angles[0])
	theta = deg2rad(angles[1])
	phi = deg2rad(angles[2])

	B = array([
		[          0,            sin(phi),             cos(phi)],
		[          0, cos(phi)*cos(theta), -sin(phi)*cos(theta)],
		[ cos(theta), sin(phi)*sin(theta),  cos(phi)*sin(theta)]
		])

	e321dotRad = B.dot(omega)
	e321dot = rad2deg(e321dotRad)
	return e321dot

###############################################################################
#
#	e313dot2omega() 
#
#	inputs:
#
#	outputs:
#
#
###############################################################################

def e313dot2omega(e321dot, angles):
	psi = deg2rad(angles[0])
	theta = deg2rad(angles[1])
	phi = deg2rad(angles[2])

	B = array([
		[         -sin(theta),         0, 1],
		[ sin(phi)*cos(theta),  cos(phi), 0],
		[ cos(phi)*cos(theta), -sin(phi), 0]
		])
	omega = B.dot(e313dot)
	return omega

###############################################################################
#
#	omega2e313dot()
#
#	inputs:
#
#	outputs:
#
#
###############################################################################


def omega2e313dot(omega, angles):
	theta1 = deg2rad(angles[0])
	theta2 = deg2rad(angles[1])
	theta3 = deg2rad(angles[2])

	B = array([
		[              sin(theta3),              cos(theta3),           0],
		[  cos(theta3)*sin(theta2), -sin(theta3)*sin(theta2),           0],
		[ -sin(theta3)*cos(theta2), -cos(theta3)*cos(theta2), sin(theta2)]
		])

	e313dotRad = B.dot(omega)
	e313dot = rad2deg(e313dotRad)
	return e321dot


###############################################################################
#
#	prv2C() 
#
#	inputs:
#
#	outputs:
#
#
###############################################################################

def prv2C(e, PHI):
	PHI = deg2rad(PHI)
	e1 = e[0]
	e2 = e[1]
	e3 = e[2]
	SIGMA = 1 - cos(PHI)
	S = SIGMA

	C = array([
		[    e1**2*S+cos(PHI), e1*e2*S+e3*sin(PHI), e1*e3*S-e2*sin(PHI)],
		[ e2*e1*S-e3*sin(PHI),    e2**2*S+cos(PHI), e2*e3*S+e2*sin(PHI)],
		[ e3*e1*S+e2*sin(PHI), e3*e2*S-e3*sin(PHI),    e3**2*S+cos(PHI)]
		])

	return C

###############################################################################
#
#	C2prv() 
#
#	inputs:
#
#	outputs:
#
#
###############################################################################

def C2prv(C):
	PHI = arccos( 0.5*(trace(C)-1) )
	eHat = array([
		C[1,2]-C[2,1],
		C[2,0]-C[0,2],
		C[0,1]-C[1,0]
		])/(2*sin(PHI))
	prv = hstack(eHat,PHI)
	return prv

###############################################################################
#
#	omega2sigmadot() 
#
#	inputs:
#
#	outputs:
#
#
###############################################################################

def omega2sigmaDot(omega, sigma):
	B = (1 - norm(sigma)**2)*identity(3) + \
		2*tilde(sigma) + \
		2*outer(sigma,sigma)
		
	sigmaDot = 0.25*B.dot(omega)
	return sigmaDot

###############################################################################
#
#	sigma2C() 
#
#	inputs:
#
#	outputs:
#
#
###############################################################################

def sigma2C(sigma):
	sigmaTilde = tilde(sigma)
	sigmaNorm = norm(sigma)
	return identity(3) + \
		(8*sigmaTilde.dot(sigmaTilde)-4*(1-sigmaNorm**2)*sigmaTilde)/\
		(1+sigmaNorm**2)**2

