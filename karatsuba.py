'''
Karatsuba matrix multiplication algorithm
By: Amit Hendin

The following program is my implementation of karatsuba matrix multiplication algorithm
'''

from num_arr_util import *

def vadd(va,vb):
	a = va.copy()
	b = vb.copy()
	makeLenEq(a,b)
	c = []
	carry = 0
	l = len(a)
	for j in range(l):
		i = l - j - 1
		s = a[i] + b[i] + carry
		carry = int(s / 10)
		c.insert(0, s % 10)
	if carry != 0:
		c.insert(0, carry)
	return c

def _karatsuba(va,vb):
	makeLenEq(va,vb)
	if len(va) == 1:
		return va[0]*vb[0]
	
	n = len(va)
	n_2 = int(n/2)

	a = va[:n_2]
	b = va[n_2:]

	c = vb[:n_2]
	d = vb[n_2:]

	ac = _karatsuba(a, c)
	bd = _karatsuba(b, d)
	a_bc_d = _karatsuba(vadd(a,b), vadd(c,d))
	ad_bc = a_bc_d - ac - bd

	s = ac*(10**n) + bd + ad_bc*(10**n_2)

	print(va,"*",vb,"=",s)

	return s

def karatsuba(na,nb):
	return _karatsuba(ntov(na), ntov(nb))