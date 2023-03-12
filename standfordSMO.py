'''
Simplied SMO
By: Amit Hendin
Date: 1/12/2022

My implementation of the simplified SMO specified in the standford article http://cs229.stanford.edu/materials/smo.pdf
'''
import numpy as np

# the linear kernel as specified in problem 10
def kernel_linear(X):
    return np.dot(X,X.transpose())

# the RBF kernel
def kernel_rbf(X, gamma):
    X_norm = np.sum(X ** 2, axis=-1)
    K = np.exp(-gamma * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)))
    return K

#input: Y=classification vector, A=lagran multipliers, K=kernel matrix, B=bias, i=index of datapoint
#output: the prediction for the data point i with the given SVM parameters
def predict(Y,A,K,B, i):
    ay = np.multiply(Y, A)
    return np.dot(ay, K[i]) + B

#input: Y=classification vector, A=lagran multipliers, K=kernel matrix, B=bias, i=index of datapoint
#output: the error in the prediction of the SVM for the given point i
def calc_error(Y,A,K,B, i):
    return predict(Y,A,K,B, i) - Y[i]

#input: Y=classification vector, A=lagran multipliers, K=kernel matrix, B=bias
#output: the index of the point in the dataset with maximal error with the given SVM
def choose_point_heuristic(Y,A,K,B):
    N = len(Y)
    #assume 0 is the point with the largest error
    max_err_point = 0
    max_err = calc_error(Y,A,K,B, 0)

    #iterate over all points in the dataset and get the one
    # with the greatest error
    for i in range(1,N):
        Ei = calc_error(Y,A,K,B, i)
        if Ei > max_err:
            max_err = Ei
            max_err_point = i

    return max_err_point

#input: X = training data points, Y = training data points' classification,
# max_passes = the amount of passes to continue without update do before exiting, aka converging
# EPS = threshold for acceptable Lagrange multiplier,
# TOL = tolerance of the error of the hyperplane
# C = margin of error of the SVM
# kernel = name of the kernel to use
# gamma = if using RBF kernel, pass in gamma value for that kernel
#output: vector W and scalar B such that W + [B] is the optimal hyperplane
# found by SMO algorithm
def SMO(X, Y, max_passes=100, EPS=10e-5, TOL=10e-3, C=1, kernel='linear', gamma=1):
    N = len(X) #number of data points
    W = np.zeros(len(X[0]))  # weights
    B = 0  # bais or "threshold"
    A = np.zeros(N)  # lagrange multipliers
    # precalculated kernel values
    if kernel == 'linear':
        K = kernel_linear(X)
    elif kernel == 'RBF':
        K = kernel_rbf(X, gamma)
    else:
        raise 'kernel ' + str(kernel) + ' not available'

    passes = 0

    #converge when we don't update for 100 iterations in a row
    while passes < max_passes:
        updated = False

        for i in range(N):
            Ei = calc_error(Y,A,K,B, i)
            if (Y[i]*Ei < -TOL and A[i] < C) or (Y[i]*Ei > TOL and A[i] > 0):
                j = choose_point_heuristic(Y,A,K,B)
                Ej = calc_error(Y,A,K,B, j)
                ai_old = A[i]
                aj_old = A[j]
                if Y[i] != Y[j]:
                    L = max(0, A[j]-A[i])
                    H = min(C, C+A[j]-A[i])
                else:
                    L = max(0, A[i]+A[j]-C)
                    H = min(C, A[i]+A[j])

                if L == H:
                    continue

                eta = 2*K[i][j] - K[i][i] - K[j][j]
                if eta >= 0:
                    continue

                A[j] = A[j] - Y[j]*(Ei-Ej)/eta
                if A[j] > H:
                    A[j] = H
                elif A[j] < L:
                    A[j] = L

                if abs(A[j] - aj_old) < EPS:
                    continue

                A[i] = A[i] + Y[i]*Y[j]*(aj_old-A[j])

                b1 = B - Ei - Y[i]*(A[i] - ai_old)*K[i][i] - Y[j]*(A[j] - aj_old)*K[i][j]
                b2 = B - Ej - Y[i]*(A[i] - ai_old)*K[i][j] - Y[j]*(A[j] - aj_old)*K[j][j]

                if A[i] > 0 and A[i] < C:
                    B = b1
                elif A[j] > 0 and A[j] < C:
                    B = b2
                else:
                    B = (b1+b2)/2

                W = W + Y[i] * (A[i] - ai_old) * X[i] + Y[j] * (A[j] - aj_old) * X[j]
                updated = True

            #if we didn't update the SVM, increase the passes so we know when we've
            #done max_passes passes without updating
            if not updated:
                passes += 1
            else: #if we've done an, update reset passes
                passes = 0

    #return the hyperplane
    return W, B