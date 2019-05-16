import numpy as np
import cvxpy as cp

def glrm(A, k, loss, reg):
    """Minimal implementation of Generalized Low-Rank Model.
    
    See Udell et al. (2016) [doi: 10.1561/2200000055].
    
    Parameters
    ----------
    A : np.ndarray [m x n]
        Data matrix with np.nan as missing values.
    k : int
        Rank of approximation. A ~ X @ Y where X [m x k], Y [k x n].
    loss, reg : functions
        Loss function and regularization function, e.g.,
        cp.sum_squares, lambda x: cp.norm(x, 1), lambda x: 0, etc.
        
    Returns
    -------
    X, Y : np.ndarray, np.ndarray
    """
    m, n = A.shape
    mask = np.isfinite(A).astype('int')
    
    M = A.copy()
    M[mask == 0] = 0
    
    X , Y  = cp.Variable ((m, k)), cp.Variable ((k, n))
    Xf, Yf = cp.Parameter((m, k)), cp.Parameter((k, n))
    
    # Initialize with SVD over known values + noise
    U, s, Vh = np.linalg.svd(M)
    Xf.value = U[:,:k] @ np.diag(np.sqrt(s[:k])) \
                + np.sqrt(1e-2 / k) * np.random.randn(m, k)
    Yf.value = np.diag(np.sqrt(s[:k])) @ Vh[:k,:] \
                + np.sqrt(1e-2 / k) * np.random.randn(k, n)
    
    prob_X = cp.Problem(cp.Minimize(
                loss(M - cp.multiply(mask, X @ Yf)) + reg(X)))
    prob_Y = cp.Problem(cp.Minimize(
                loss(M - cp.multiply(mask, Xf @ Y)) + reg(Y)))
    
    lst_X = prob_X.solve(solver=cp.SCS) * 10
    lst_Y = prob_Y.solve(solver=cp.SCS) * 10
    
    for _ in range(1000): # max iterations for alternating minimization
        cur_X = prob_X.solve(solver=cp.SCS, eps=1e-2, max_iters=100)
        Xf.value = X.value
        
        cur_Y = prob_Y.solve(solver=cp.SCS, eps=1e-2, max_iters=100)
        Yf.value = Y.value
        
        if np.abs((cur_X - lst_X) / lst_X) < 1e-2 and \
            np.abs((cur_Y - lst_Y) / lst_Y) < 1e-2:
            break
        lst_X, lst_Y = cur_X, cur_Y
    
    return X.value, Y.value

if __name__ == '__main__':
    import warnings

    A = np.array(
        [[ 1,  2,  3,  4],
         [ 2,  4,  6,  8],
         [ 3,  6,  9, 12],
         [ 4,  8, 12, 16]]).astype('float')
    
    A[0,0] = np.nan
    A[1,1] = np.nan
    A[2,2] = np.nan
    A[3,3] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, Y = glrm(A, 1, cp.sum_squares, lambda x: 0)
        
    print('Original matrix')
    print(A)

    print()
    print('Imputed matrix')
    print(X @ Y)
