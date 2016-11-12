"""FIT module

"""

import numpy as np
import sys
from numpy import linalg
from matplotlib import pyplot as plt



def fjh(f,x,p,epsilon=1e-6,small=1e-10):
    """function, jacobian, and hessian
    f,J,H = fjh(f,x,p)
    
for a function of the form
    y = f(x,p)
    
    y in R
    x in R^m
    p in R^n
    
Return the value at f(x,p), and the Jacobian and Hessian with respect to
perturbations in H.  Requries x and p to be numpy ndarray types.
"""
    n = p.size
    J = np.matrix(np.zeros((n,1)))
    H = np.matrix(np.zeros((n,n)))
    # compute a perturbation vector
    dp = np.maximum(np.abs(p)*epsilon,small)
    if n==1:
        f00 = f(x,p)
        f10 = f(x,p+dp)
        f20 = f(x,p-dp)
        J[0,0] = 0.5*(f10-f20)/dp
        H[0,0] = (f10 - 2.*f00 + f20)/dp/dp
    else:
        ptest = p.copy()
        # Set up Fval as an n dimensional 3x3x... tensor
        # It will contain the explicit values of the function evaluation
        # findex is a n-element list corresponding to the indices
        # [0,0,...] corresponds to the function evaluation at p
        # [0,1,...] corresponds to the function evaluation with p[1]+dp[1]
        # [0,-1,...] corresponds to the function evaluation with p[1]-dp[1]
        # This approach prevents redundant function calls while evaluating
        # the Hessian
        findex = n*[0]
        Fval = np.zeros(n*(3,))
        # initial evaluation
        Fval[tuple(findex)] = f(x,ptest)
        # positive and negative perturbations
        for pi in range(n):
            findex[pi] = -1
            ptest[pi] = p[pi] - dp[pi]
            Fval[tuple(findex)] = f(x,ptest)
            findex[pi] = 1
            ptest[pi] = p[pi] + dp[pi]
            Fval[tuple(findex)] = f(x,ptest)
            # double perturbations for the hessian
            for pj in range(pi+1,n):
                findex[pj] = 1
                ptest[pj] = p[pj] + dp[pj]
                Fval[tuple(findex)] = f(x,ptest)
                # return p and findex to their original values
                findex[pj] = 0
                ptest[pj] = p[pj]
            findex[pi] = 0
            ptest[pi] = p[pi]

        # findex should already be zeros, but let's be sure.
        findex = n*[0]
        f00 = Fval[tuple(findex)]
        # Calculate the Jacobian and Hessian for this point
        # Loop through each of the parameters
        # Then loop through paired parameters to calculate the hessian
        for pi in range(n):
            findex[pi] = 1
            f10 = Fval[tuple(findex)]
            findex[pi] = -1
            f20 = Fval[tuple(findex)]
            J[pi,0] = 0.5*(f10 - f20)/dp[pi]
            H[pi,pi] = (f10 - 2.*f00 + f20)/dp[pi]/dp[pi]
            findex[pi] = 0
            for pj in range(pi+1,n):
                findex[pj] = 1
                f01 = Fval[tuple(findex)]
                findex[pi] = 1
                f11 = Fval[tuple(findex)]
                H[pi,pj] = (f11 - f10 - f01 + f00)/dp[pi]/dp[pj]
                H[pj,pi] = H[pi,pj]
                findex[pj] = 0
                findex[pi] = 0
    return f00, J, H


class fit:
    """Nonlinear curve fit
    
    F = fit( x=(x,d,a,t,a,...), 
             y=(y,d,a,t,a,...), 
             f=myfun, 
             p=(my,param,eters))
    
F is a "fit object" which holds data, intermediate calculations, and 
results for a nonlinear curve fit.  The function "myfun" is a of the 
form

    y = f(x,p)
    
where y is a scalar, x is a scalar or a 1d array, and p is a scalar or a
1d array.

The y-data array is a 1D numpy array containing a series of measurements
made at independent variable conditions in the x-data array.  If the x-
data array is 1D, it must be the same length as y.  Calls to f will be

    f(x[index],p)
    
If x-data is 2D, each row will correspond to an element of y.  Calls to 
f will be of the form

    f(x[index,:],p)

The values passed to the "p" variable will be treated as initial guesses
for p.  Scalar values, lists, tuples, and arrays will automatically be 
formatted appropriately.

>>> import numpy as np
>>> def f(x,p):
>>>     return np.exp(p*x)
>>> x = np.arange(0,1,.02)
>>> y = 1. - np.exp(-4.*x)
>>> F = fit(x, y, f, -5., a=-.9, b=.9)

Once the fit is performed, it can be called like a function
>>> F(.3)
0.69880578808779781

Parameters are stored explicitly
>>> F.P
array(-3.9999999999999996)
>>> F.a
-1.0
>>> F.b
1.0

The behavior of the FIT object is configured by a series of optional 
parameters. 

->  a, b
When a and/or b are specified, they add a scale and offset respectively
to the fit, so that the fit is of the form
    b + a*f
    
->  w
The w keyword allows the user to specify an array of weighting factors.
The w array must be the same length as y.

->  xarray
When the xarray keyword is set True, it indicates that the function, f, 
accepts multidimensional arrays for x

->  usefjh
The usefjh keyword should be provided a funciton that can be used in 
place of the built-in fjh() function to evaluate the Jacobian and 
Hessian matrices.  The call signature should appear
    (fi, Ji, Hi) = fjh(f, xi, p)

f is the function
xi is the x data array or scalar
p is the parameter array
fi, Ji, and Hi are the scalar, Jacobian matrix, and Hessian matrix of
f with respect to the elements of p.  If p is an n-element vector, then
Ji should be an (n,1) numpy matrix and Hi should be an (n,n) numpy 
matrix.

There are a number of intermediate calculations that are stored in the
object:

F._e2   is the variance of the quantity (yi - f(xi,p))
F._COV  is the covariance matrix
F._LAM  is the Lambda matrix
F._R    is the residual vector
F._PSI  is the jacobian self-product matrix

"""

    def __init__(self,x, y, f, p,  w=None, 
                     a=None, b=None, xarray=False, usefjh=None):

        epsilon = 1e-6
        small = 1e-10
        # Set up the defaults
        self.X = None       # X data
        self.Y = None       # Y data
        self.W = None       # weighting factors
        self.P = None       # parameters
        self.a = None
        self.b = None
        self._fjh = None
        self._conv = False  # convergence flag
        self._xarray = False # array compatibility flag
        self._e2 = None     # y error variance
        self._LAM = None    # Solution matrix
        self._PSI = None    # Jacobian self-product
        self._R = None      # Residual vector
        self._COV = None    # Coefficient covariance matrix
        self._dp = None     # parameter step vector
        
        # Test for vector support
        self._xarray = bool(xarray)
        # Test for fjh override
        if usefjh != None:
            self._fjh = usefjh

        # Test the a and b parameters
        if a!=None:
            self.a = float(a)
        if b!=None:
            self.b = float(b)
        
        # Force Y to be a 1D array and detect the number of points, N
        self.Y = np.array(y)
        N = self.Y.size
        self.Y.shape = (N,)

        # Set up the X array
        # If x elements are declared individually, parse them
        self.X = np.array(x)
        # check the number of elements
        if self.X.shape[0]!=N:
            raise Exception('X data length does not match Y.')
        if self.X.ndim==2:
            m = self.X.shape[1]
            # Eliminate any redundant dimensions
            if m == 1:
                self.X.shape = (N,)
        elif self.X.ndim>2 or self.X.ndim<1:
            raise Exception('X must be 1 or 2 dimensional.')

        # Stow the p vector and do some size checking
        self.P = np.array(p)
        n = self.P.size
        # if P is a scalar, eliminate the redundant dimension
        if n>1:
            self.P.shape = (n,)
        else:
            self.P.shape = ()

        # Force W to be a 1D weighting factor array
        if w!=None:
            self.W = np.array(w)
            if self.W.size != N:
                raise Exception('Number of weighting factors does not match the size of Y')
            self.W.shape = (N,)
        else:
            self.W = np.ones((N,))

        # Test the function
        try:
            if self.X.ndim==2:
                xtest = self.X[0,:]
            else:
                xtest = self.X[0]
            out = f(xtest,self.P)
        except:
            print "Failed while testing the fit function."
            raise Exception(sys.exc_info()[1])
        self.f = f


        ########################
        # Solve the fit
        ########################
        Rnow = None
        pv_flag = self.P.ndim==1
        for count in range(200):
            Rlast = Rnow
            self._step() # Update _LAM and _R to current P
            Rnow = linalg.norm(self._R)
            # Test the residual vector
            # its magnitude should be shrinking
            if Rlast==None or Rlast>Rnow:
                # Solve for the step vector
                self._dp = np.asarray(linalg.solve(self._LAM, -self._R))
                if pv_flag:
                    self.P += self._dp[:n,0]
                else:
                    self.P += self._dp[0,0]
                pi = n
                if self.b !=None:
                    self.b += self._dp[pi,0]
                    pi += 1
                if self.a !=None:
                    self.a += self._dp[pi,0]
                # Test for convergence
                if np.all(np.abs(self._dp) < np.maximum(self.P*epsilon,small)):
                    # Update the residuals and matrix one last time
                    self._step()
                    self._COV = self._LAM.I * self._COV * self._LAM.I * self._e2
                    self._conv = True
                    return
            # If the residual magnitude is growing
            else:
                # go backwards a half-step
                self._dp *= .5
                if pv_flag:
                    self.P -= self._dp[:n,0]
                else:
                    self.P -= self._dp[0,0]
                pi = n
                if self.b !=None:
                    self.b -= self._dp[pi,0]
                    pi += 1
                if self.a !=None:
                    self.a -= self._dp[pi,0]
        raise Exception('Failed to converge while solving for coefficients.')
        
        

    def __call__(self,x):
        """Evaluate the fit function"""
        if not isinstance(x,np.ndarray):
            x = np.array(x)
        # Check for conformity with the data dimensions
        m = self.xdim()
        # If the function requires vectors
        if m>1:
            if x.ndim < 1:
                raise Exception('This function requires an x-vector with "{:d}" elements'.format(m))
            # If this is a 2-dimensional array
            elif x.ndim == 2:
                # Check that the columns match xdim
                if x.size[1]!=m:
                    raise Exception(
'The x array contains {:d} columns, but the function requires {:d}.'.format(x.shape[1],m))
                # If arrays are supported
                if self._xarray:
                    y = self.f(x,self.P)
                # If arrays are not supported, step it out
                else:
                    y = np.zeros((x.shape[0],))
                    for xi in range(x.shape[0]):
                        y[xi] = self.f(x[xi,:],self.P)
            # If x is a 1D array
            elif x.ndim == 1:
                # Check that its dimension matches xdim
                if x.size[0]!=m:
                    raise Exception(
'This function requires an x-vector with "{:d}" elements'.format(m))
                y = self.f(x,self.P)
            # If x is higher than dimension 2
            else:
                raise Exception('Cannot process dimension data arrays with dimension greater than 2.')
        # If the function requires scalars
        else:
            if x.ndim==0 or self._xarray:
                y = self.f(x,self.P)
            elif x.ndim==1:
                y = np.zeros((x.size,))
                for xi in range(x.size):
                    y[xi] = self.f(x[xi],self.P)
            else:
                raise Exception(
'Function requires scalars; cannot parse arrays with dimension higher than 1.')
        # Apply linear scale and offset
        if self.a != None:
            y *= self.a
        if self.b != None:
            y += self.b
        return y


    def _step(self, epsilon=1e-5, small=1e-10):
        """Updates the _LAM and _R members
    FIT._step()

_R is the residual vector containing the gradient of the square of error.
"""
        
        N = self.N()
        nt = self.ntot()
        n = self.np()
        m = self.xdim()
        
        xv_flag = self.X.ndim==2
        pv_flag = self.P.ndim==1
        
        self._LAM = np.matrix(np.zeros((nt,nt)))
        self._COV = np.matrix(np.zeros((nt,nt)))
        self._R = np.matrix(np.zeros((nt,1)))
        self._e2 = 0.
        # Initialize the jacobian and hessian matrices
        J = np.matrix(np.zeros((nt,1)))
        H = np.matrix(np.zeros((nt,nt)))

        # Loop through the x points
        for xi in range(N):
            if xv_flag:
                x = self.X[xi,:]
            else:
                x = self.X[xi]

            # Evaluate the function and its derivatives
            fi,Ji,Hi = self._fjh(self.f,x,self.P)
            
            # Handle the offset and scaling coefficient specially
            pi = nt-1
            if self.a != None:
                H[:n,:n] = self.a * Hi
                H[pi,:n] = Ji
                H[:n,pi] = Ji
                J[:n] = self.a * Ji
                J[pi] = fi
                fi *= self.a
                pi -= 1
            else:
                H[:n,:n] = Hi
                J[:n] = Ji
                
            if self.b != None:
                J[pi] = 1
                fi += self.b
            
            # Build up the solution matrix and residuals vector
            dy = f00 - self.Y[xi]
            PSI = J*J.T
            self._COV += PSI * self.W[xi] * self.W[xi]
            self._LAM += (PSI + dy*H)*self.W[xi]
            self._R += dy*J*self.W[xi]
            self._e2 += dy*dy
        self._e2 /= N

                
    
    def N(self):
        """Return the number of points in the fit
    N = FIT.N()
 See also: np(), ntot(), xdim()
"""
        return self.Y.size
        
        
    def np(self):
        """Return the number of parameters in the fit excluding a and b
    n = FIT.np()
    
To include a and b, use ntot()   
 See also: N(), ntot(), xdim()
"""
        return self.P.size
        
        
    def ntot(self):
        """Return the total number of parameters in the fit (including a and b)
    n = FIT.ntot()

To exclude a and b, use n()
 See also: N(), np(), xdim()
"""
        return self.P.size + (self.a!=None) + (self.b!=None)

    
    def xdim(self):
        """Return the dimension of x or number of columns in X
    m = FIT.xdim()
    
 See also: N(), np(), ntot()
"""
        if self.X.ndim==2:
            return self.X.shape[1]
        else:
            return 1


    def cov(self):
        """Return a coefficient covariance matrix
    C = FIT.cov()

When the b and a coefficients are in use, they are represented in the 
last columns/rows.  For example, in an example where one scalar 
parameter, p, is in use with a and b, the covariance matrix will appear

        / Cpp Cpb Cpa \
    C = | Cbp Cbb Cba |
        \ Cap Cab Caa /
        
where Cxy = covar(x,y).  The a and b rows and columns will be omitted 
when a or b is not in use.
"""
        if self._COV != None:
            return self._COV.copy()
        else:
            raise Exception('The covariance matrix has not been calculated!')




    def plot(self,fig=None):
        """Produce a plot of the data and fit
        
"""
        if fig==None:
            f = plt.figure()
            ax = f.add_subplot(111)
        else:
            f = plt.figure(fig)
            f.clf()
            ax = f.add_subplot(111)
            
        if self.xdim() > 1:
            raise Exception('No support for multi dimensional plots.')

        xmin = self.X.min()
        xmax = self.X.max()
        xr = xmax - xmin
        xmin -= xr * 0.1
        xmax += xr * 0.1
        x = np.arange(xmin,xmax,xr*.05)
        y = self(x)
        
        ax.plot(self.X,self.Y,'ko')
        ax.plot(x,y,'k')
