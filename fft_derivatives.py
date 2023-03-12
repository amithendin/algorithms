'''
Simple usage demo of the FFT algorithm for multiplying vectors
'''
from cmath import *
import math
from fft import *

def eval(p, x):
	sum = 0
	x_p = 1
	for i in range(0, len(p)):
		sum += p[i]*x_p
		x_p = x_p*x
	return sum

def derive(p):
	d_p = []
	for i in range(1, len(p)):
		d_p.append(p[i]*i)
	return d_p

def eval_der(p, x):
	E = []
	while len(p) > 0:
		E.append( eval(p, x) )
		p = derive(p)
	return E

def eval_der_fft_half(p, x):
	n = len(p)
	
	A_tag = []
	for i in range(0,n):
		A_tag.append( x + exp((2j * pi * i) / n) );

	D = []
	for i in range(0,n):
		D.append( eval(p, A_tag[i]) ) 

	B = iFFT(D)

	E = [ B[0] ]
	for i in range(1,n):
		curr = i * E[i-1] * B[i] / B[i-1]
		E.append( curr )

	round_complex_arr(A)
	round_complex_arr(D)
	round_complex_arr(B)
	round_complex_arr(E)

	print('A ', A)
	print('D ', D)
	print('B ', B)

	return E

x0 = 3
A = [5,1,4,1]

def s(r,n,a,x):
	sum = 0
	for j in range(0,n):
		fj = a[j]*math.factorial(j)
		gj = pow(x, j-r)/math.factorial(j-r) if j > r else 0
		sum = sum + fj*gj
	return sum

def eval_der_fft(a, x):
	n = len(a)
	
	sr = 0
	for j in range(0,n):
		fj = a[j]*math.factorial(j)
		gj = pow(x, j-r)/math.factorial(j-r) if j > r else 0
		sr = sr + fj*gj
	fr = math.factorial(r)

	D = []	
	for r in range(0,n):
		D.append( exp((2j * pi * r) / n) * sr / fr );

	B = iFFT(D)

	E = [ B[0] ]
	for i in range(1,n):
		curr = i * E[i-1] * B[i] / B[i-1]
		E.append( curr ) 

	round_complex_arr(B)
	round_complex_arr(E)

	print('B\'', B)

	return E

eval_der_fft_half(A, x0)

print('fft eddition', eval_der_fft(A, x0) )
print('Standard', eval_der(A, x0) )
