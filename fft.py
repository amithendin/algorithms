'''
Fast Fourier Transfrom algorithm
By: Amit Hendin

The following program is my implementation of Fast Fourier Transfrom algorithm
'''
from cmath import *

def round_complex(c):
	return round(c.real, 2) + round(c.imag, 2) * 1j

def round_complex_arr(arr):
	for i in range(0,len(arr)):
		arr[i] = round(arr[i].real, 2) + round(arr[i].imag, 2) * 1j

#FTT of vector A
def FFT(a, d=1):
	n = len(a)
	
	if n == 1:
		return a

	wn = exp(  (d * 2j * pi) / n )

	a_even = list()
	a_odd = list()
	
	for i in range(0,n):
		if i % 2 == 0:
			a_even.append(a[i])
		else:
			a_odd.append(a[i])

	y_even = FFT(a_even, d)
	y_odd = FFT(a_odd, d)
	
	y = [0j]*n
	w = 1
	for k in range( 0, int(n/2) ):
		y[k] = y_even[k] + w * y_odd[k]
		y[k + int(n/2)] = y_even[k] - w * y_odd[k]
		w = w*wn

	return y

#inverse FFT of vector a
def iFFT(a):
	n = len(a)
	
	if n == 0:
		return a

	y = FFT(a, -1)
	for i in range(0,n):
		y[i] /= n

	return y

#returns the convolution between vectors A and B
def conv(A,B):
	n = len(A)

	zeros = [0]*n

	A_p = FFT(A + zeros)
	B_p = FFT(B + zeros)
	
	C_p = []
	for i in range(2*n):
		C_p.append( A_p[i]*B_p[i] )

	C = iFFT(C_p)

	return C

print ('--------------\nresult', FFT([-1,-3,2,1]),'\n--------------')
print ( round_complex( exp((-2j*pi)/1 ) ) )
