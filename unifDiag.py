# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:46:31 2017

@author: amorvan
"""
from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
            
                
def unifDiag(matrix, tol = 1e-4):
    """
    Parameters:
    -matrix: square and symmetric matrix (should be checker outside)
    -tol: tolerance parameter to be equal to a certain value
    
    This function makes uniform the diagonal of the symmetric square matrix
    in parameter.
    
    It returns the matrix "diagonal uniformization" (matrix_out) and the orthogonal
    matrix built to do it (rotation) from the matrix in parameter (matrix_in) s.t.
    
    matrix_out = np.dot(rotation, np.dot(matrix_in,rotation.T))
    
    """

    matrix_cp = np.copy(matrix)
    dim = matrix_cp.shape[0]
    
    # loops initialization
    rotation = np.eye(dim)
    MeanDiag = np.trace(matrix_cp)/dim
    # indices
    #iInf = find(np.diag(matrix_cp) < MeanDiag-tol)
    iInf = [ i for i,coef in zip(xrange(dim), np.diag(matrix_cp)) if coef < MeanDiag-tol]
    #iSup = find(np.diag(matrix_cp) > MeanDiag+tol)
    iSup = [ i for i,coef in zip(xrange(dim), np.diag(matrix_cp)) if coef > MeanDiag + tol]
    nIter = 0
    
    # loops
    while nIter< (dim-1) and len(iInf) != 0 and len(iSup) != 0:
        
        m = iInf[0]
        n = iSup[0]
        
        # Givens rotation parameters computation
        a = matrix_cp[m,m]
        b = matrix_cp[n,m]
        d = matrix_cp[n,n]
        
        rac = math.sqrt( b**2+((a-d)/2)**2 )
        
        c1 = ( (a-d)/2 )/ rac
        s1 = (    b    )/ rac
        c2 = ( MeanDiag - (a+d)/2 )/ rac
        s2 = math.sqrt(1 - c2**2)    # should not be complex if a<MeanDiag<d
        c = math.sqrt( (1 + c1*c2 - s1*s2 )/2 )
        s = -(c1*s2 + s1*c2)/(2*c)
        
        # matrix update
        row_m = np.copy(matrix_cp[m,:])
        row_n = np.copy(matrix_cp[n,:])   
                
        matrix_cp[m,:] = c*row_m - s*row_n
        matrix_cp[n,:] = s*row_m + c*row_n
        matrix_cp[:,m] = matrix_cp[m,:]
        matrix_cp[:,n] = matrix_cp[n,:]
        matrix_cp[m,m] = MeanDiag
        matrix_cp[n,n] = a+d - MeanDiag
        matrix_cp[m,n] = -s2 * rac
        matrix_cp[n,m] = matrix_cp[m,n]
        
        # rotation update
        col_m = np.copy(rotation[:, m])
        col_n = np.copy(rotation[:,n])
        rotation[:,m] = c*col_m - s*col_n
        rotation[:,n] = s*col_m + c*col_n
        
        # indices sets update
        #iInf = find(np.diag(matrix_cp) < MeanDiag-tol)
        iInf = [ i for i,coef in zip(xrange(dim), np.diag(matrix_cp)) if coef < MeanDiag-tol]
        #iSup = find(np.diag(matrix_cp) > MeanDiag+tol)
        iSup = [ i for i,coef in zip(xrange(dim), np.diag(matrix_cp)) if coef > MeanDiag + tol]
        nIter = nIter + 1
        
#        print('nIter = ' + str(nIter))
#        print('iInf = ' + str(iInf))
#        print('iSup = ' + str(iSup))
        
    return matrix_cp, rotation







def test_unifDiag():
    """
    Test the function UnifDiag.
    """
    # compute the square symmetric matrix Ai
    N = 32
    Ai = np.random.normal(0.0, 1.0, (N, N))
    Ai = Ai + Ai.T
    #Ai = np.dot(Ai, Ai.T) # (for a positive Ai)
    A, rotation = unifDiag(Ai,1e-4)    

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('before dediagonalization')
    cax1 = ax1.imshow(Ai, cmap=cm.coolwarm)  
    fig1.colorbar(cax1, ticks=[-1, 0, 1])
    fig1.show()    
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('after dediagonalization -> \ndiagonal coef. should be uniform.')
    cax2 = ax2.imshow(A, cmap=cm.coolwarm)
    fig2.colorbar(cax2, ticks=[-1, 0, 1])
    fig2.show()
              
    print('R A R.T - Ai', np.max(np.max(np.abs(np.dot(rotation, np.dot(A, rotation.T)) - Ai)))) # should be small
    print('U U.T - I', np.max(np.max(np.abs(np.dot(rotation, rotation.T) - np.eye(N))))) # symmetric
    print('A-A.T', np.max(np.max(np.abs(A - A.T))))
    
    print(np.diag(A))


def test_impact_where_R():
    """
    Function to show how to apply R (the rotation given by UnifDiag)
    to obtain uniform diagonal coefficients on rotated data.
    """
    
    n = 100
    c = 32
    
    print('TRY 1, X is n x c')
    X = np.random.normal(0.0, 1.0, (n, c))
    CovV = np.dot( X.T, X)
    assert CovV.shape == (c,c)
    CovV_goal, R = unifDiag(CovV,1e-4)
    
#    X_unifDiag = np.dot(X, R.T) # NOT WORKING
    X_unifDiag = np.dot(X, R) # WORKS
    CovV_unifDiag = np.dot(X_unifDiag.T, X_unifDiag)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('TRY1, before dediagonalization')
    cax1 = ax1.imshow(CovV, cmap=cm.coolwarm)  
    fig1.colorbar(cax1, ticks=[-1, 0, 1])
    fig1.show()    
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('TRY1, after dediagonalization -> \ndiagonal coef. should be uniform.')
    cax2 = ax2.imshow(CovV_unifDiag, cmap=cm.coolwarm)    
    fig2.colorbar(cax2, ticks=[-1, 0, 1])
    fig2.show()    

    print('TRY 2, X is c x n')
    X = np.random.normal(0.0, 1.0, (c, n))
    CovV = np.dot( X, X.T)
    assert CovV.shape == (c,c)
    CovV_goal, R = unifDiag(CovV,1e-4)
    
    X_unifDiag = np.dot(R.T, X) # WORKS
#    X_unifDiag = np.dot(R, X) # NOT WORKING
    CovV_unifDiag = np.dot(X_unifDiag, X_unifDiag.T)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('TRY 2, before dediagonalization')
    cax1 = ax1.imshow(CovV, cmap=cm.coolwarm)  
    fig1.colorbar(cax1, ticks=[-1, 0, 1])
    fig1.show()    
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('TRY3, after dediagonalization -> \ndiagonal coef. should be uniform.')
    cax2 = ax2.imshow(CovV_unifDiag, cmap=cm.coolwarm)    
    fig2.colorbar(cax2, ticks=[-1, 0, 1])
    fig2.show()   



    
    
if __name__ == '__main__':

#    test_unifDiag()
    test_impact_where_R()