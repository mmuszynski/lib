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
from numpy import sqrt, argmax, trace, array, zeros
from numpy.linalg import norm
def dcm2quaternion(dcm):
    tr = trace(dcm)
    b2 = array(
	  [(1 + tr)/4,
       (1 + 2*dcm[0, 0] - tr)/4,
       (1 + 2*dcm[1, 1] - tr)/4,
       (1 + 2*dcm[2, 2] - tr)/4
       ])
    case = argmax(b2)
    b = b2
    if case == 0:
        b[0] = sqrt(b2[0])
        b[1] = (dcm[1, 2] - dcm[2, 1])/4/b[0]
        b[2] = (dcm[2, 0] - dcm[0, 2])/4/b[0]
        b[3] = (dcm[0, 1] - dcm[1, 0])/4/b[0]
    elif case == 1:
        b[1] = sqrt(b2[1])
        b[0] = (dcm[1, 2] - dcm[2, 1])/4/b[1]
        if b[0] < 0:
            b[1] = -b[1]
            b[0] = -b[0]
        b[2] = (dcm[0, 1] + dcm[1, 0])/4/b[1]
        b[3] = (dcm[2, 0] + dcm[0, 2])/4/b[1]
    elif case == 2:
        b[2] = sqrt(b2[2])
        b[0] = (dcm[2, 0] - dcm[0, 2])/4/b[2]
        if b[0] < 0:
            b[2] = -b[2]
            b[0] = -b[0]
        b[1] = (dcm[0, 1] + dcm[1, 0])/4/b[2]
        b[3] = (dcm[1, 2] + dcm[2, 1])/4/b[2]
    elif case == 3:
        b[3] = sqrt(b2[3])
        b[0] = (dcm[0, 1] - dcm[1, 0])/4/b[3]
        if b[0] < 0:
            b[3] = -b[3]
            b[0] = -b[0]
        b[1] = (dcm[2, 0] + dcm[0, 2])/4/b[3]
        b[2] = (dcm[1, 2] + dcm[2, 1])/4/b[3]
    return b

def dcm2mrp(dcm):
    b = dcm2quaternion(dcm)
    q = array([
        b[1] / (1 + b[0]),
        b[2] / (1 + b[0]),
        b[3] / (1 + b[0])
    ])
    return q

def mrp2dcm(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    qm = norm(q)
    d1 = qm*qm
    S = 1 - d1
    d = (1 + d1)*(1 + d1)
    C = zeros((3, 3))
    C[0, 0] = 4*(2*q1*q1 - d1) + S*S
    C[0, 1] = 8*q1*q2 + 4*q3*S
    C[0, 2] = 8*q1*q3 - 4*q2*S
    C[1, 0] = 8*q2*q1 - 4*q3*S
    C[1, 1] = 4*(2*q2*q2 - d1) + S*S
    C[1, 2] = 8*q2*q3 + 4*q1*S
    C[2, 0] = 8*q3*q1 + 4*q2*S
    C[2, 1] = 8*q3*q2 - 4*q1*S
    C[2, 2] = 4*(2*q3*q3 - d1) + S*S
    C = C / d
    return C
