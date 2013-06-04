import os
import sys
import re
import math
from spacecharge import Grid3D

class Field_Parser3D:
	""" 3D field parser """

	def __init__(self):
		""" Create instance of the Field_Parser3D class """
		self.__lines =  []
	
	def __del__(self):
		del self.__lines
		
	def getLimits(self,filename):
		
		infile = open(filename,"r")
		# 		Resize the grid so more files can be processed
		numLines = 0
		xmin,xmax,ymin,ymax,zmin,zmax = 0,0,0,0,0,0
		for line in infile.readlines():
			splitline = line.split()
			value = map(float, splitline)
			
			# finding minimums and maximums of data to compute range
			xmax = max(xmax,value[0])
			ymax = max(ymax,value[1])
			zmax = max(zmax,value[2])
			xmin = min(xmin,value[0])
			ymin = min(ymin,value[1])
			zmin = min(zmin,value[2])
			
			numLines += 1
		
		xmax = max(xmax,0)
		ymax = max(ymax,0)
		zmax = max(zmax,0)
		xmin = min(xmin,0)
		ymin = min(ymin,0)
		zmin = min(zmin,0)
		
		print "Min and Max values: " , xmin, xmax, ymin, ymax, zmin, zmax, numLines, "\n" 

		limits = [xmin, xmax, ymin, ymax, zmin, zmax, numLines]
			
		return limits	
	#limits is a list of format [xmin,xmax,ymin,ymax,zmin,zmax]
	def getRange(self,limits):
		xmin = limits[0]
 		xmax = limits[1]
 		ymin = limits[2]
 		ymax = limits[3]
 		zmin = limits[4]
 		zmax = limits[5]
 		
 	
 		Xrange = xmax-xmin
		Yrange = ymax-ymin
		Zrange = zmax-zmin
		
		range = [Xrange,Yrange,Zrange]
		print "Range of X,Y,Z values in data: ", Xrange, " ", Yrange, " ", Zrange
		
		return range


###############################################################################
#  gets the gridsize give the range of each variable 
# and the step of each variable
##############################################################################	
	def getGridSize(self,range, step, usrLimits):
		
		for i in xrange(3):
			range[i] = range[i]*1.0/step[i]		
		gridSize = [range[0]+1,range[1]+1, range[2]+1]

		
		xrnge = usrLimits[1] - usrLimits[0]
		yrnge = usrLimits[3] - usrLimits[2]
		zrnge = usrLimits[5] - usrLimits[4]
		usrRange = [xrnge,yrnge,zrnge]
		
		for i in xrange(3):
			usrRange[i] = (usrRange[i]*1.0/step[i]) + 1
		for i in xrange(3):
			if(usrRange[i]<gridSize[i]): 
				gridSize[i] = usrRange[i]
		gridSize = map(int,gridSize)
		print "Grid Size [x,y,z]: " , gridSize
		

		return gridSize
###############################################################################
#Returns the coordinates in the grid given the rawNumbers 
#and the limits of each variable.
############################################################################## 	
	def getCoordinates(self, gridSize, step,rawNumbers, limits):
		
		coordinates = [rawNumbers[0] ,rawNumbers[1],rawNumbers[2]]
		for i in xrange(len(coordinates)):
			coordinates[i] = coordinates[i]*(1.0/step[i])
			coordinates[i] = coordinates[i]-limits[2*i]/step[i]
		coordinates = map(int, coordinates)
		
		return coordinates
	
	
#####
# Checks to see if the given coordinates are within the range specified
#####
	def checkLimits(self, arrayLimits, value):
		if(value[0] >= arrayLimits[0] and 
		value[0] <= arrayLimits[1]):
			if(value[1] >= arrayLimits[2] and 
			value[1] <= arrayLimits[3]):
				if(value[2] >= arrayLimits[4] and 
				value[2] <= arrayLimits[5]):
					return True
		else:
			return False
		
	def checkGrid(self,step,value):
		localStep = [0,0,0]
		localValue = [0,0,0]
		for i in xrange(3):
			localStep[i] = 2*step[i]
			localValue[i] = 2*value[i]
		map(int, localStep)
		map(int, localValue)
		
		for i in xrange(3):
			if(value[i]%step[i] != 0):
				return False
		else:
			return True
		

	
	
###############################################################################
# Parameters 
# filename: name of the text file to be processed
# xsize: size of the grid in the x diminsion
# ysize: size of the grid in the y diminsion
# zsize: size of the grid in the z diminsion
# All Grid sizes are user defined.
###############################################################################
	def parse(self, filename, xmin,xmax,ymin,ymax,zmin,zmax,xstep,ystep,zstep):
		
		usrLimits = [xmin,xmax,ymin,ymax,zmin,zmax]		
		limits = self.getLimits(filename)
		
		range = self.getRange(limits)
		step = [xstep,ystep,zstep]
		
		
		gridSize = self.getGridSize(range, step, usrLimits)
	
 		numLines = limits[6]	
 		
		print "Number of lines in the file: ",numLines , "\n"
		
		#for now we will say that the size of the grid encompasses all datapoints
		print "GridSize " , gridSize[0],gridSize[1],gridSize[2]
  		fieldgrid3DBx = Grid3D(gridSize[0],gridSize[1],gridSize[2])
  		fieldgrid3DBy = Grid3D(gridSize[0],gridSize[1],gridSize[2])
  		fieldgrid3DBz = Grid3D(gridSize[0],gridSize[1],gridSize[2])
  		fieldgrid3DMag = Grid3D(gridSize[0],gridSize[1],gridSize[2])
  		xGrid = Grid3D(gridSize[0],gridSize[1],gridSize[2])
  		yGrid = Grid3D(gridSize[0],gridSize[1],gridSize[2])
  		zGrid = Grid3D(gridSize[0],gridSize[1],gridSize[2])
  		
  		
  		
#   		setGridX = (usrLimits[0],usrLimits[1])
#   		setGridY = (usrLimits[2],usrLimits[3])
#   		setGridZ = (usrLimits[4],usrLimits[5])
  		
  		
	
		print
	
		##
		# Maps values from file to grid.
		##
		infile1 = open(filename,"r")
		x =1
		for line in infile1.readlines():
			splitLine = line.split()
			rawNumbers = map(float, splitLine)
# 			Maps data points to integers so that they can be evaluated for stepsize
			testRS = map(int, rawNumbers)
#   			print rawNumbers
  			if(self.checkGrid(step,rawNumbers) and
 			  self.checkLimits(usrLimits,rawNumbers) 
 			  ):
			  	
# 			  	print gridSize, step, rawNumbers, limits
				coordinates = self.getCoordinates(gridSize,step,rawNumbers, usrLimits)
#   				print coordinates
# 				print rawNumbers
					
 				xGrid.setValue(rawNumbers[0]/100.0, coordinates[0], coordinates[1], coordinates[2])
 		 		yGrid.setValue(rawNumbers[1]/100.0, coordinates[0], coordinates[1], coordinates[2])
 		 	 	zGrid.setValue(rawNumbers[2]/100.0, coordinates[0], coordinates[1], coordinates[2])
		 	 	fieldgrid3DBx.setValue(rawNumbers[3]/10000.0, coordinates[0], coordinates[1], coordinates[2])		 	
		 		fieldgrid3DBy.setValue(rawNumbers[4]/10000.0, coordinates[0], coordinates[1], coordinates[2])
		 	
 				fieldgrid3DBz.setValue(rawNumbers[5]/10000.0, coordinates[0], coordinates[1], coordinates[2])
 			 	getMag = ((rawNumbers[3]**2.0+rawNumbers[4]**2.0+rawNumbers[5]**2.0)**0.5)/10000.0
 		 	
 	 			fieldgrid3DMag.setValue(getMag, coordinates[0], coordinates[1], coordinates[2])
#   		 		if (x<1000):
#  				  	print "At Coordinates " , coordinates, " x == ", x
#  				 	print "Values X,Y,ZGrid ",  xGrid.getValueOnGrid(coordinates[0],coordinates[1],coordinates[2]), yGrid.getValueOnGrid(coordinates[0],coordinates[1],coordinates[2]), zGrid.getValueOnGrid(coordinates[0],coordinates[1],coordinates[2])
#  				 	print "Values fieldGrids " ,  fieldgrid3DBx.getValueOnGrid(coordinates[0],coordinates[1],coordinates[2]), fieldgrid3DBy.getValueOnGrid(coordinates[0],coordinates[1],coordinates[2]), fieldgrid3DBz.getValueOnGrid(coordinates[0],coordinates[1],coordinates[2]) 
 				 	
#     				x += 1
 		  		 
 		  		 
 		MagList = [fieldgrid3DBx,fieldgrid3DBy,fieldgrid3DBz,fieldgrid3DMag,xGrid,yGrid,zGrid]
 		
		return MagList
	