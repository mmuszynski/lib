#! /usr/bin/env python3
# H+
#	Title   : orbits.py
#	Author  : Matt Muszynski
#	Date    : 09/04/16
#	Synopsis: Functions for orbit stuff
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

#------------------------------------------------------------------
#
# day2sec()
#
#------------------------------------------------------------------

def day2sec(timeInDays):
	timeInSeconds = timeInDays*24*3600
	return timeInSeconds

#------------------------------------------------------------------
#
# day2sec()
#
#------------------------------------------------------------------

def sec2day(timeInSeconds):
	timeInDays = timeInSeconds/24/3600
	return timeInDays


#------------------------------------------------------------------
#
# timeConvert() converts times between different formats. I copied
#	the idea from LASP, but I coded it up. Makes liberal use of 
#	Python's time module.
#
#	Inputs:
#		time: the actual time (as number or string) to be converted
#		format_from: the time format of the time passed in
#		format_to: the time format we're going to.
#
#	Outputs:
#		time: a list, string, or decimal as appropriate.
#
#	Supported time formats:
#		UTC:
#		JD:
#		YDNHMS: [2016,279,14,40,32.333]
#
#------------------------------------------------------------------

def timeConvert(time, format_from, format_to):
	from datetime import datetime
	from datetime import timedelta
	import re
	import pdb

	format_from = format_from.upper()
	format_to = format_to.upper()

	#--------------------------------------------------------------
	#
	#	If the user passed in something that's not a list, turn
	#		it into one. That way we can treat it the same as a
	#		list passed by the user when we do conversions below.
	#
	#--------------------------------------------------------------

	if not(isinstance(time,list)):
		time = [time]

	#ydnhms has to work a little differently.
	if format_from == 'YDNHMS' and not(isinstance(time[0],list)):
		time = [time] 


	#--------------------------------------------------------------
	#
	#	This section takes the time and converts it to a Python
	#		time object.
	#
	#
	#--------------------------------------------------------------

	if format_from == 'UTC':
		for i in range(0,len(time)):
			time[i] = \
			datetime.strptime(time[i], '%Y/%jT%H:%M:%S.%f')

	if format_from == 'JD':
		for i in range(0,len(time)):
			#We make a delta time object. It is then added to the
			#JD epoch of 12h Jan 1, 4713 BC to create a time object
			#for the time in question.

			#2415020.5 is JD for Jan 1, 1900. I chose this because
			#datetime doesn't support BCE years, and I can't get
			#all the way back to the JD epoch. 1900 is convenient
			#since I indend to use this for astrodynamics, not
			#astronomy. This puts the epoch before any notion
			#of leap seconds.
			deltatime = timedelta(time[i]-2415020.5)
			jan_1_1900 = datetime.strptime( \
				'1900/001T00:00:00.00', '%Y/%jT%H:%M:%S.%f')
			time[i] = jan_1_1900 + deltatime

	if format_from == 'YDNHMS':
		for i in range(0,len(time)):
			#user can pass in arrays shorter than necessary for the
			#code. This fills in zeros in that case.
			if len(time[i]) == 1:
				time[i] = [time[i][0],1]
			if len(time[i]) == 2:
				time[i] = [time[i][0],time[i][1],0]
			if len(time[i]) == 3:
				time[i] = [time[i][0],time[i][1],time[i][2],0]
			if len(time[i]) == 4:
				time[i] = [time[i][0],time[i][1],time[i][2],time[i][3],0]

			#prepare the seconds value. It's a little harder since
			#it could have decimals or not, and the value before
			#the decimal could be 1 or 2 digits.
			seconds = str(time[i][4]) + '.'
			split_seconds = seconds.split('.')
			seconds = \
				('0' + str(split_seconds[0]))[-2:] + \
				'.' + \
				str(split_seconds[1])

			if seconds[-1:] == '.':
				seconds = seconds + '0'

			utc_string = \
				str(time[i][0]) + '/' + \
				('00' + str(time[i][1]))[-3:] + 'T' + \
 				('0' + str(time[i][2]))[-2:] + ':' + \
				('0' + str(time[i][3]))[-2:] + ':' + \
				seconds
			time[i] = \
				datetime.strptime(utc_string, '%Y/%jT%H:%M:%S.%f')

	#--------------------------------------------------------------
	#
	#	This section takes the universal standard and converts it
	#		to the output format.
	#
	#--------------------------------------------------------------

	if format_to == 'UTC':
		for i in range(0,len(time)):
			time[i] = time[i].strftime('%Y/%jT%H:%M:%S.%f')
	
	if format_to == 'JD':
		jan_1_1900 = datetime.strptime( \
				'1900/001T00:00:00.00', '%Y/%jT%H:%M:%S.%f')
		for i in range(0,len(time)):
			delta = time[i] - jan_1_1900
			delta_jd = delta.days + delta.seconds/3600./24. \
				+ delta.microseconds/3600./24./100000.
			#using 1/1/1900 as pseudo-epoch.
			time[i] = 2415020.5 + delta_jd

	if format_to == 'YDNHMS':
		for i in range(0,len(time)):
			time[i] = time[i].strftime('%Y/%jT%H:%M:%S.%f')
			utc_split = re.split('/|T|:',time[i])
			#the regex leaves things as strings, this floats them.
			time[i] = [float(j) for j in utc_split]

	if format_to == 'CALENDAR':
		for i in range(0,len(time)):
			time[i] = time[i].strftime('%b %d, %Y %H:%M:%S.%f')

	if len(time) == 1:
		time = time[0]

	return time

























