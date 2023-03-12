'''
Some utility functions used in other files here
'''

from math import *

def ntov(n, base=10):
	tmp = n
	v = []
	while tmp > 0:
		v.append(tmp % base)
		tmp = int(tmp/base)
	return v 

def vton(v, base=10):
	n = 0
	for i in range(0, len(v)):
		n += v[i]*(base**i)
	return n 

def extendToNearest2Pow(a):
	nearest2Pow = (2**ceil(log(len(a))/log(2) ))
	a.extend([0]*(nearest2Pow - len(a)))

def makeLenEq(a,b):
	if len(a) > len(b):
		if len(a) % 2 != 0:
			a.extend([0])
		b.extend( [0]*(len(a) - len(b)) )
	elif len(b) > len(a):
		if len(b) % 2 != 0:
			b.extend([0])
		a.extend( [0]*(len(b) - len(a)) )
