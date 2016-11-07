# Numerical tools for PYro
# These include tools to analyze and invert functions
# 
# Methods intended for external use:
#   newton1()   1-dimensional newton solver
#   newton2()   2-dimensional newton solver
#   bis1()      1-dimensional bisection solver
#   bis2()      2-dimensional bisection solver
#
# Classes
#   SimplexN    N-dimensional simplex (parent class)
#   Simplex2    2-dimensional simplex (child class)

import numpy as np


class Vertex:
    """Vertex       A point in x-space and values in f-space
    V = Vertex(f,X)
    
X is an N-tuple or array representing a point, X, in N-space.
f is a function assumed to be a mapping from N-space to N-space.  In
other words, f should be a function that operates like this:
 Y = f(X)
where X is an N-element vector and Y is an N-element vector.

V.X     stores a copy of the X-vector
V.Y     stores a copy of the result of a call to f(X)
V.f     is a pointer to the function, f
"""
    def __init__(self,f,X):
        self.X = np.array(X)
        self.f = f
        self.Y = np.array(f(X))
        
    def __repr__(self):
        out = "<Vertex ("
        for x in self.X[:-1]:
            out += repr(x)+","
        out += (repr(self.X[-1]) + ")=>(")
        for y in self.Y[:-1]:
            out += repr(y)+","
        out += (repr(self.Y[-1]) + ")>")
        return out
    
    



class SimplexN:
    """SimplexN     An N-dimensional simplex
    SN = SimplexN(...)
    
"""

    def __init__(self,copy=None,V=None):
        # Initialize the standard members
        self.N = 0
        self.V = []
        self._edges = None
        self._faces = None
        self._children = None
        self._parent = None
        self._longest = None
        # Case out the arguments
        if copy:
            self.N = copy.N
            self.V = list(copy.V)
            if self._edges:
                self._edges = list(copy._edges)
            if copy._faces:
                self._faces = list(copy._faces)
            self._longest = copy._longest

        elif X:
            self.V = list(X)
            self.N = len(self.V)-1
        # Determine if this is a face or a complete simplex
        # Faces will be of lower dimension than their member vertices
        self._isface = (self.N<len(self.V[0].X))


    def __repr__(self):
        return "<{:d}-Simplex>".format(self.N)


    def __getitem__(self,index):
        return self.V[index]
        
    def __in__(self,x):
        return x in self.V

    def _make_edges(self):
        """Form new edges from the vertices already in the simplex"""
        # Let each edge be indicated by the vertices it includes
        self._edges = []
        for ii in range(self.N):
            for jj in range(ii+1,self.N+1):
                self._edges.append(
                    Edge(X=(self.V[ii],self.V[jj])))
        
        
    def _make_faces(self):
        """Form new faces from the vertices and edges already in the simplex"""
        # Let each face be indicated by the vertex it excludes
        self._faces = []
        for ii in range(self.N+1):
            # Create an N-1 simplex excluding vertex ii
            this = SimplexN(X=(self.V[:ii]+self.V[ii+1:]))
            self._faces.append(this)
            #The edges for each face will be all edges NOT including vertex ii
            this._edges = []
            index = 0
            for jj in range(self.N):
                for kk in range(jj+1,self.N+1):
                    if jj!=ii and kk!=ii:
                        this._edges.append(self._edges[index])
                    index+=1


    def _test(self,error=False,message="",ihist=[]):
        """_test    Runs diagnostics on the simplex
    SN._test()
    
(I) In order to pass the diagnostics, the following must be true:
(I.1) There must be N+1 vertices
(I.2) There must be N*(N+1)/2 edges
(I.3) All ii,jj = _etov(index) must agree with the vertices in each edge

(II) If faces are present, 
(II.1) Each face must also pass this test
(II.2) Each face must be order N-1
(II.3) Face ii must contain all vertices EXCEPT SN.V[ii]
(II.4) Each edge of each face must belong to SN (accidental redundant 
        edges?)

(III) If children are present
The simplex is a face if the vertices are of dimension more than the 
simplex order.
(III.1) There can be two or zero children
(III.2) If the simplex is not a face, children must also pass this test
(III.3) The parent of each child must be SN
(III.4) Each child must be order N
(III.5) All but one child vertex must be in SN
(III.6) The new vertex must be the midpoint of SN's longest edge.
"""

        isface = (self.N < len(self.V[0].X))
        def preface():
            out = "IHIST: " + repr(ihist) + "\nFace: "
            if isface:
                out+="True\n"
            else:
                out+="False\n"
            return out
                
        # (I) Edge-vertex diagnostics
        # (I.1) There must be N+1 vertices
        if len(self.V)!=self.N+1:
            error = True
            message += preface() + \
"""Failed (I.1) {0:d}-Simplex with {1:d} vertices\n\n""".format(self.N,len(self.V))
            # Halt the test here.
            return error,message        
        
        # (I.2) There must be N*(N+1)/2 edges
        if len(self._edges) != self.N*(self.N+1)/2:
            error = True
            message += preface() + \
"""Failed (I.2) {0:d}-Simplex with {1:d} edges\n\n""".format(self.N,len(self._edges))
            # Halt the test here.
            return error,message

        # (I.3) Edges must map to correct vertices
        index = 0
        for ii in range(self.N):
            for jj in range(ii+1,self.N+1):
                E = set((self._edges[index].V[0], self._edges[index].V[1]))
                S = set((self.V[ii],self.V[jj]))
                
                if E != S:
                   error = True
                   message += preface() + \
"""Failed (I.3) Edge {:d} does not map to vertices {:d},{:d}\n\n""".format(index,ii,jj)
                index += 1
        
        # (II) If faces are present
        if self._faces:
            for index in range(len(self._faces)):
                this = self._faces[index]
                # (II.1) All faces must pass this test.
                error,message = this._test(error=error,message=message,ihist=ihist)
                # (II.2) All faces must be order N-1
                if this.N != self.N-1:
                    error = True
                    message += preface() + \
"""Failed (II.2) Face {:d} is order {:d}, but should be {:d}\n\n""".format(index,this.N,self.N-1)
                # (II.3) Face must contain all points except index
                D = set(self.V) - set(this.V)
                if len(D)!=1 or self.V[index] not in D:
                    error = True
                    message += preface() + \
"""Failed (II.3) Face {:d} contains the incorrect or redundant vertices\n\n""".format(index)
                # (II.4) Face edges must match the parent's
                if set(this._edges) - set(self._edges):
                    error = True
                    message += preface() + \
"""Failed (II.4) Face {:d} contains an edge not in the master simplex\n\n""".format(index)

        # III Test children
        if self._children:
            # III.1 There may only be two children
            if len(self._children)!=2:
                error = True
                message += preface() + \
"""Failed (III.1) A simplex may only have two children.\n\n"""
            for index in (0,1):
                this = self._children[index]
                # III.2 Children of simplexes that aren't faces must pass this test
                if not isface:
                    error,message = this._test(
                        error=error, message=message, ihist = ihist + [index])
                
                # III.3 Each child's parent must be self
                if this.parent is not self:
                    error = True
                    message += preface() + \
"""Failed (III.3) Child {:d}'s parent is not correct.\n\n""".format(index)
    
                # III.4 Each child must be an N-simplex
                if this.N != self.N:
                    error = True
                    message += preface() + \
"""Failed (III.4) Child {:d} is an {:d}-simplex, but the parent is an {:d}-simplex.\n\n""".format(index,self._children[0].N,self.N)

                CV = set(this.V)
                V = set(self.V)
                new = CV - V
                # III.5 All vertices must be identical save one
                if len(new)!=1:
                    error = True
                    message += preface() + \
"""Failed (III.5) Child {:d} vertices do not match its parent's.\n\n""".format(index)

                # III.6 The new vertex must be the midpoint of the longest edge
                # (test is only valid when III.5 passes)
                elif new.pop() is not self.longest().midpoint():
                    error = True
                    message += preface() + \
"""Failed (III.6) Child {:d}'s new vertex does not appear to be correct.\n\n""".format(index)
        return error,message



    def _vtoe(self,ii,jj):
        """Convert vertex indices to an edge index
    index = SN._vtoe(ii,jj)
    
Returns an index such that SN._edges[index] is an edge connecting 
Vertices SN.V[ii] and SN.V[jj]. In order to work properly, ii MUST 
BE LESS THAN jj.
"""
        # This code was removed to reduce overhead
        # This method is not intended for users, and errors should be caught
        # in the implementation.
#        if jj<ii:
#            temp = jj
#            jj = ii
#            ii = temp
#        elif jj==ii:
#            raise Exception("SIMPLEXN: No edge connects a vertex to itself.")
        return (ii * (2*self.N-1-ii))/2 + jj - 1


    def _etov(self,index):
        """Convert edge index to vertex indices
    ii,jj = SN._etov(index)
    
Returns indices such that SN._edges[index] is an edge connecting 
Vertices SN.V[ii] and SN.V[jj]. In order to work properly, index MUST BE
LESS THAN OR EQUAL TO SN.N
"""
        # increment the index so counting starts at zero
        index+=1
        for ii in range(self.N,-1,-1):
            if index<=ii:
                ii = self.N-ii
                jj = index + ii
                return ii,jj
            index -= ii
        raise Exception("SIMPLEXN: edge index is out of range.")
        
        # This algorithm was replaced with the one above for efficiency
        # While the one below is purely arithmetic, it requires five 
        # multiplications and a square root.  The one above only uses
        # order N additions and subtractions.
        #temp = 1+2*self.N
        #ii = int((temp - np.sqrt(temp*temp-8*(index-1)))/2)
        # These lines were removed to reduce overhead
        # This method is not intended for users, and errors should be caught
        # in the implementation
#        if ii>=self.N:
#            raise Exception("SIMPLEXN: Edge index out of range.")
        #jj = index - ii*(2*self.N-1-ii)/2
        return ii,jj


    def isface(self):
        """isface       Is this the face of a bigger simplex?
    TorF = SN.isface()
"""
        


    def longest(self):
        """longest()    return the longest edge
    e = SN.longest()

The longest edge is a persistent property.
"""
        if not self._longest:
            L = 0.
            for index in range(len(self._edges)):
                temp = self._edges[index].xlength()
                if temp>L:
                    self._longest = index
                    L = temp
        return self._edges[self._longest]


    def bisect(self):
        """bisect()     bisect the simplex
    SN.bisect()
        
Populates the children members.  SN.children is a list 
containing two SimplexN objects that are bisections of the original 
SimplexN.  SN.bisx is the new Vertex that is the bisection of the 
longest edge.

The bisection is persistent.
"""
        # If the bisection has already been performed, return the children
        if self._children:
            return self._children
        
        e = self.longest()  # Longest edge
        ei = self._longest  # lonest edge index
        ai,bi = self._etov(ei)  # Get the vertices and face indices
        ea,eb = e.bisect()  # bisect the edge
        # If the vertices are in reverse order of the _etov indices
        # swap the bisection results so ea will contain self.V[ai]
        if e.X[0] != self.V[ai]:
            temp = eb
            eb = ea
            ea = temp
        vm = e.midpoint()   # retreive the midpoint

        # Create two child simplexes as copies of the parent
        A = SimplexN(copy=self)
        A._parent = self
        B = SimplexN(copy=self)
        B._parent = self
        # Modify one vertex of each
        # Notice that in simplex A, vertex bi is replaced and ai is kept
        # In simplex B, the opposite is true
        A.V[bi] = vm
        B.V[ai] = vm
        # Modify the edges of each
        # First, place the bisected edges in place of the original edge
        # Note that we were careful that ea and eb weren't mixed up (above)
        A._edges[ei] = ea
        B._edges[ei] = eb
        # Now, create the new edges
        # loop through all vertices, but skip the ones that already exist
        for ii in range(self.N+1):
            if ii not in (ai,bi):
                # If this vertex isn't either of the verticies forming the
                # bisected edge, then it will form a new edge to the midpoint
                # of the bisected edge. (vm)
                new_edge = Edge( V=(vm,self.V[ii]) )
                # In A, the new vertex is in index bi, so form new edges there.
                if ii<bi:
                    index = A._vtoe(ii,bi)
                else:
                    index = A._vtoe(bi,ii)
                A._edges[index] = new_edge
                # in B, the new vertex is in ai
                if ii<ai:
                    index = B._vtoe(ii,ai)
                else:
                    index = B._vtoe(ai,ii)
                B._edges[index] = new_edge
        # Now we need to form bisected faces.
        
        
        
class Edge (SimplexN):
    """A 1-simplex with special functions for
    (1) Persistent linear bisection
    (2) Persistent calculation of length
"""


    def __init__(self,*varg,**kwarg):
        SimplexN.__init__(self,*varg,**kwarg)
        self._xlength = None
        self._midpoint = None

    def __repr__(self):
        return "<Edge " + repr(self.V[0]) + "," + repr(self.V[1]) + ">"

    def xlength(self):
        """Return the length of the edge in x-space"""
        if not self._xlength:
            self._xlength = np.linalg.norm(self.V[1].X - self.V[0].X)
        return self._xlength

    def midpoint(self):
        """Return a vertex at the midpoint of the edge"""
        if not self._midpoint:
            self._midpoint = Vertex(self.V[0].f, 
                                    0.5*(self.V[0].X+self.V[1].X))
        return self._midpoint
        
    def bisect(self):
        """Return the two edges resulting from a bisected edge
    ea,eb = e.bisect()
    
ea contains e.X[0] and eb contains e.X[1].  Both contain the midpoint."""
        if not self._children:
            vm = self.midpoint()
            self._children = [Edge(X=(self.V[0],vm)), 
                              Edge(X=(vm,self.V[1]))]
        return self._children



def newton1(f,xinit=0.,fval=0.,ep=1e-6,small=1e-10,N=100,debug=False):
    """newton1      1-dimensional Newton solver
    x = newton1(f)

For some function, f(x), one dimensional Newton iteration seeks the 
value x, such that f(x)==0.  The behavior of newton1 is configurable
with keyword arguments:

xinit (default: 0.)
This scalar number determines the initial guess for x.  The default of
zero will rarely be a good choice, so it will become important to
establish a reasonable first guess.  Using PYro's p_def and T_def
parameters are reasonable starting points.

fval (default: 0.)
This scalar number is an alternative value to which f(x) should 
converge.  In other words, if
    fval == f(x)
then x is a solution.

ep  (default: 1e-6)
Short for "epsilon" this indicates the fractional precision to which 
f should converge and x should be perturbed to estimate a derivative.
f is a solution if 
    abs(f-fval) <= min(ep*fval, small)
See the "small" parameter for more information.
The derivative of f(x) is approximated by
    dx = max(abs(x*ep),small)
    df = f(x+dx) - f(x)
    dfdx = df / dx

small (default: 1e-10)
Specifies a value for a "small" number.  Since ep (epsilon) is a 
fractional precision, the numerical precision and perturbation vainish
when f or x is close to zero.  Specifying a numerically small number 
fixes that problem.

N (default: 100)
Runaway iteration limit.  If the iteration repeats this number of times
then throw an error and exit rather than iterating infinitely.

debug (default:False)
When true, newton1 will print a history of the guesses to stdout.
"""

    small=abs(small)

    # There are four state variables in this algorithm:
    # x: The current guess for x
    # fx: The value of f(x) at x
    # ex: The energy of f(x) or f^2(x)
    # dx: The delta x under test (change in x from old value)

    x = xinit # use the initial guess
    ex = float('inf') # inf forces passing the first quality check
    dx = 0.
    # convergance has occurred when ex < ex_test
    ex_test = max(ep*ep*fval*fval, small*small)    
    # iterate no more than N times
    count = 0
    while count < N:
        # First shift x and ex into "old" values
        x_old = x
        ex_old = ex
        # Evaluate the new f(x)
        x = x_old+dx
        fx = f(x)
        count += 1
        if debug:
            print "({:d}) x={:f},f={:f}".format(count,x,fx)
        ex = (fx-fval)
        ex = ex*ex
        # Test the quality of the guess
        while ex > ex_old and count < N:
            # If the guess is a little wild, reel it back in until
            # we get something that seems a little more reasonable
            dx /= 2.
            x = x_old+dx
            fx = f(x)
            count += 1
            if debug:
                print "({:d}) x={:f},f={:f}".format(count,x,fx)
            ex = fx-fval
            ex = ex*ex
    
        # Test for convergence
        if ex < ex_test:
            return x

        # Calculate the new dx
        px = max(small,x*ep) # perturbation in x
        dfdx = (f(x+px) - fx)/px
        dx = (fval-fx)/dfdx
    
    # If we've made it to this point in the code, we failed.
    raise Exception('NEWTON1: Failed to converge.')
        

def newton2(f,g,xinit=0.,yinit=0.,fval=0.,gval=0.,ep=1e-6,small=1e-8,N=100,debug=False):
    """newton2      2-dimensional Newton solver
    x = newton2(f,g)

For function, f(x,y) and g(x,y), Newton iteration seeks the values x 
and y, such that f(x,y)==0 and g(x,y)==0.  The behavior of newton2 is 
configurable with keyword arguments:

xinit and yinit (default: 0.)
This scalar number determines the initial guesses for x and y.  The 
default of zero will rarely be a good choice, so it will become 
important to establish a reasonable first guess.  Using PYro's p_def 
and T_def parameters are reasonable starting points.

fval and gval (default: 0.)
This scalar number is an alternative value to which f() and g() should 
converge.  In other words, if
    fval == f(x)
then x is a solution.

ep  (default: 1e-6)
Short for "epsilon" this indicates the fractional precision to which 
f should converge and x should be perturbed to estimate a derivative.
f is a solution if 
    abs(f-fval) <= min(ep*fval, small)
See the "small" parameter for more information.
The derivatives are approximated by
    dx = max(abs(x*ep),small)
    df = f(x+dx) - f(x)
    dfdx = df / dx

small (default: 1e-10)
Specifies a value for a "small" number.  Since ep (epsilon) is a 
fractional precision, the numerical precision and perturbation vainish
when f or x is close to zero.  Specifying a numerically small number 
fixes that problem.

debug (default:False)
When true, newton1 will print a history of the guesses to stdout.
"""

    small=abs(small)

    # There are four state variables in this algorithm:
    # x: The current guess for x
    # fx: The value of f(x) at x
    # ex: The energy of f(x) or f^2(x)
    # dx: The delta x under test (change in x from old value)

    x = xinit # use the initial guesses
    y = yinit
    ex = float('inf') # inf forces passing the first quality check
    dx = 0.
    dy = 0.
    # convergance has occurred when ex < ex_test
    ex_test = max(ep*ep*fval*fval, small*small)
    # iterate no more than N times
    count = 0
    while count < N:
        # First shift x and ex into "old" values
        x_old = x
        y_old = y
        ex_old = ex
        # Evaluate the new f and g
        x = x_old+dx
        y = y_old+dy
        fxy = f(x,y)
        gxy = g(x,y)
        count += 1
        if debug:
            print "({:d}) x={:f},y={:f},f={:f},g={:f}".format(count,x,y,fxy,gxy)
        df = fval-fxy
        dg = gval-gxy
        ex = df*df + dg*dg
        # Test the quality of the guess        df = fval - fxy
        while ex > ex_old and count < N:
            # If the guess is a little wild, reel it back in until
            # we get something that seems a little more reasonable
            dx /= 2.
            dy /= 2.
            x = x_old+dx
            y = y_old+dy
            fxy = f(x,y)
            gxy = g(x,y)
            count += 1
            if debug:
                print "({:d}) x={:f},y={:f},f={:f},g={:f}".format(count,x,y,fxy,gxy)
            df = fval - fxy
            dg = gval - gxy
            ex = df*df + dg*dg
    
        # Test for convergence
        if ex < ex_test:
            return x,y

        # Calculate the new dx and dy
        px = max(small,x*ep) # perturbation in x
        py = max(small,y*ep)
        x1 = x+px
        y1 = y+py
        dfdx = (f(x1,y) - fxy)/px   # approximate the jacobiandx = (fval-fx)/dfdx
        dfdy = (f(x,y1) - fxy)/py
        dgdx = (g(x1,y) - gxy)/px
        dgdy = (g(x,y1) - gxy)/py
        dd = dfdx*dgdy - dfdy*dgdx
        # df and dg are borrowed from the last step of the error calculation
        dx = (df * dgdy - dg * dfdy)/dd # invert the jacobian
        dy = (-df * dgdx + dg * dfdx)/dd
    
    # If we've made it to this point in the code, we failed.
    raise Exception('NEWTON2: Failed to converge.')
        


def bis1(f,xa,xb,fval=0.,ep=1e-6,small=1e-10,N=100,debug=False):
    """bis1     One-dimensional bisection routine
    x = bis1(f,xa,xb)

Seeks a value, x, such that f(x)==0.  The algorithm begins with a 
search domain between xa and xb.  The behavior of bis1 is configurable
through keyword arguments:

This scalar number determines the initial guess for x.  The default of
zero will rarely be a good choice, so it will become important to
establish a reasonable first guess.  Using PYro's p_def and T_def
parameters are reasonable starting points.

fval (default: 0.)
This scalar number is an alternative value to which f(x) should 
converge.  In other words, if
    fval == f(x)
then x is a solution.

ep  (default: 1e-6)
Short for "epsilon" this indicates the fractional precision to which 
f should converge and x should be perturbed to estimate a derivative.
f is a solution if 
    abs(f-fval) <= min(ep*fval, small)
See the "small" parameter for more information.
The derivative of f(x) is approximated by
    dx = max(abs(x*ep),small)
    df = f(x+dx) - f(x)
    dfdx = df / dx

small (default: 1e-10)
Specifies a value for a "small" number.  Since ep (epsilon) is a 
fractional precision, the numerical precision and perturbation vainish
when f or x is close to zero.  Specifying a numerically small number 
fixes that problem.

debug (default:False)
When true, newton1 will print a history of the guesses to stdout.
"""
    fa = f(xa)
    fb = f(xb)
    dfa = fa - fval
    dfb = fb - fval
    testa = (dfa>0)
    testb = (dfb>0)
    if not (testa ^ testb):
        raise Exception('BIS1: The domain does not appear to contain a solution.')
    elif testb:
        temp = fa
        fa = fb
        fb = temp
        temp = xa
        xa = xb
        xb = temp
        
    xc = 0.5*(xa+xb)
    for count in range(N):
        fc = f(xc)
        if fc>fval:
            fa = fc
            xa = xc
        else:
            fb = fc
            xb = xc

        xc = 0.5*(xa+xb)        
        if abs(xa-xb) < max(small,ep*xc):
            return xc
    raise Exception('BIS1: Failed to converge to the precision requested in {:d} steps.'.format(N))