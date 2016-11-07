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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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
    Simplexes should not be created directly from the class initializer.
Instead, use the simplex() function, which carefully populates the 
entire face and sub-face data tree.  This structure is essential for
methods like longest() and bisect() to work correclty.
"""

    def __init__(self,copy=None,V=None):
        # Initialize the standard members
        self.N = -1
        self.V = []
        self._faces = []
        # Children and parent refer a bisection tree
        self._children = None
        self._parent = None
        # The following members are for persistence of the methods that are
        # their namesakes
        self._longest = None
        self._midpoint = None
        # Case out the arguments
        if copy is not None:
            # NEED TO ADD COPY
            # This is the routine for copying a simplex to a new simplex
            pass
            return
        
        if V is not None:
            self.V = list(V)
            self.N = len(self.V)-1
            self._faces = [None for v in self.V]



    def __repr__(self):
        return "<{:d}-Simplex>".format(self.N)

        
    def __contains__(self,S):
        """Test for containment
Containment is determined by whether a simplex belongs to the tree of
a larger simplex.  

    A in B
    
Returns "True" if and only if all of the vertices of A are also vertices
of B.  Note that for B to contain A, B.N >= A.N.  If B.N == A.N, and B 
contains A, then A and B are identical, but are not necessarily part of
the same data tree.
"""
        for vv in S.V:
            if vv not in self.V:
                return False
        return True
        
        
    def _nCr(self,n,r):
        """N-choose-R function
    count = SN._nCr(n,r)

count is the integer number of possible combination of r objects from a
set of n objects"""
        out = 1
        for aa in range(r+1,n+1):
            out *= aa
        for aa in range(1,n-r+1):
            out /= aa
        return out


    def _addface(self,face):
        """Add a face to the proper face index.
    S._addface(F)

When S is an N simplex, F must be an N-1 simplex, containing only
vertices already in S.  F will be added to the _faces list at the index
such that the vertex in the V list at the same index is NOT in F.

For computational efficiency, _addface() does not require prior use of
__contains__().  Any simplex may be passed as the face argument.  If the
face does not belong to self, it will not be added to the data tree.
"""

        # Only add N-1 simplexes
        if face.N!=self.N-1:
            return

        # Initialize the face index to an invalid value
        index = -1
        # Loop over the vertices in self.
        for ii in range(self.N+1):
            # If one of them does not belong to the face
            if self.V[ii] not in face.V:
                # If this is the first vertex not belonging to the face
                if index<0:
                    index = ii
                # If a non-member vertex has already been discovered
                else:
                    # Then the face does not belong to self
                    return
        # If the candidate simplex is N-1 and only one vertex of self
        # doesn't belong to face, then face belongs to self.
        self._faces[index] = face

        

    def _test(self,error=False,message="",ihist=[]):
        """_test    Runs diagnostics on the simplex data tree
    error,message = SN._test()
    
error       boolean indicating whether there was an error
message     detailed descriptions of errors encountered

(I) The self test:
(I.1) There must be N+1 unique vertices
(I.2) All vertices must be the same dimension, but it need not be N

(II) The face tests
These are only applied for simplexes greater than order 1
(II.1) There must be N+1 unique faces
(II.2) Each face must be an order N-1 simplex
(II.3) Each face must also pass this test
(II.4) A face _faces[I] must contain all vertices except V[I]

(III) Linking tests
These are tests based on the "level" sets of simplexes.  Level sets are
compilations of all unique subordinate simplexes in the tree belonging 
to a simplex.
(III.1) The zero level set must match the vertex set
(III.2) The K-level set must contain N-choose-K unique simplexes

(IV) Children
These tests ensure that the bisection tree is properly organized.
(IV.1) If children are present, each must be order N
(IV.2) Each child must contain the correct vertices
        The vertices should be identical to the parent, except...
        longest().V[1] will be replaced by longest().midpoint() in
        _children[0]
        longest().V[0] will be replaced by longest().midpoint() in
        _children[1]
(IV.3) Each child must pass this test
(IV.4) Each child must have self as a parent
"""
        def preface():
            out = "ID history: " + repr(ihist + [id(self)]) + "\n"
            return out
        
        Vset = set(self.V)
        # (I) Self test
        # (I.1) There must be N+1 vertices
        if len(self.V)!=self.N+1:
            error = True
            message += preface() + \
"""Failed (I.1) {0:d}-Simplex with {1:d} vertices\n\n""".format(self.N,len(self.V))
            # This error is fatal
            return error,message
            
        # (I.2) All vertices must be the same dimension
        M = len(self.V.X)
        for vv in self.V:
            if len(vv.X)!=M:
                error = True
                message += preface() + \
"""Failed (I.2) Vertex x-dimensions do not match."""
            if len(vv.Y)!=M:
                error = True
                message += preface() + \
"""Failed (I.2) Vertex f(x)-dimensions do not match."""
        
        # (II) Face tests
        if self.N>1:
            # (II.1) There must be N+1 unique faces
            if len(self._faces)!=self.N+1:
                error = True
                message += preface() + \
"""Failed (II.1) {0:d}-Simplex with {1:d} faces\n\n""".format(self.N,len(self._faces))

            for I in range(self.N+1):
                this = self._faces[I]
                # (II.2) Each face must be an order N-1 simplex
                if not isinstance(this,SimplexN):
                    error = True
                    message += preface() + \
"""Failed (II.2) Found a face that is not a SimplexN\n\n"""
                    # This error is fatal
                    return error,message
                if this.N!=self.N-1:
                    error = True
                    message += preface() + \
"""Failed (II.2) A {0:d}-Simplex is a face of a {0:d}-Simplex\n\n""".format(this.N,self.N)
                # (II.3) Each face must also pass this test
                try:
                    error,message = this._test(
                        error=error, message=message, ihist=ihist + [id(self)])
                except:
                    error = True
                    message += preface() + \
"""Failed (II.3) Encountered an unexpected error while testing member faces\n\n"""
                    return error,message
                
                # (II.4) face _faces[I] must contain all vertices except V[I]
                if not error:
                    FV = set(this.V)
                    V = set(self.V)
                    V.remove(self.V[I])
                    if FV != V:
                        error = True
                        message += preface() + \
"""Failed (II.4) A face at index I must contain all vertices except V[I]\n\n"""

        # (III) Link tests
        # (III.1) level 0 must match the vertex set
        if self.level(0) != set(self.V):
            error = True
            message += preface() + \
"""Failed (III.1) Vertices do not match the level 0 set\n\n"""
        # (III.2) K level set must contain N+1-choose-K+1 unique elements
        for K in range(2,self.N):
            nlevel = len(self.level(K))
            nCr = self._nCr(self.N,K)
            if self._nCr(self.N+1,K+1) != len(self.level(K)):
                error = True
                message += preface() + \
"""Failed (III.2) Level {:d} of this {:d}-simplex contains {:d} simplexes not {:d}.\n\n""".format(K,self.N,nlevel,nCr)

        # (IV) Child tests
        if self._children:
            # (IV.1) Children must be order N simplexes
            if not isinstance(self._children[0],SimplexN) or \
                    not isinstance(self._children[1],SimplexN):
                error = True
                message += preface() + \
"""Failed (IV.1) Found a child with a data type other than SimplexN\n\n"""
                # This error is fatal
                return error,message
            if self._children[0].N!=self.N or \
                    self._children[1].N!=self.N:
                error = True
                message += preface() + \
"""Failed (IV.1) Children are not the correct order\n\n"""
            
            # (IV.2) Child k must contain longest()[k]
            LE = self.longest()
            V = list(self.V)
            bi = V.index(LE.V[1])
            V[bi] = LE.midpoint()
            if set(V) != set(self._children[0].V):
                error = True
                message += preface() + \
"""Failed (IV.2) Child 0 does not have the correct vertices.\n\n"""
            V = list(self.V)
            ai = V.index(LE.V[0])
            V[ai] = LE.midpoint()
            if set(V) != set(self._children[1].V):
                error = True
                message += preface() + \
"""Failed (IV.2) Child 1 does not have the correct vertices.\n\n"""
            # This error is fatal
            return error,message
        
            # (IV.3) Children must pass this test
            for this in self._children:
                try:
                    error,message = this._test(
                        error=error, message=message, ihist=ihist + [id(self)])
                except:
                    error = True
                    message += preface() + \
"""Failed (IV.3) Encountered an unexpected error while testing children\n\n"""
                    return error,message
            
            # (IV.4) Each child must have self as a parent
            for child in self._children:
                if child._parent != self:
                    error = True
                    message += preface() + \
"""Failed (IV.4) Child does not reference self as a parent\n\n"""
        return error,message


    def level(self,K):
        """Return a level set for the simplex
    level_set = SN.level(K)

Returns the set of unique K-simplexes belonging to SN.
"""
        # If K corresponds to the faces of this simplex
        # Just return a copy of their set
        if self.N-1==K:
            if self.N>1:
                return set(self._faces)
            else:
                return set(self.V)
        # If K is burried in the tree
        elif self.N-1>K:
            # Descend into the tree recursively
            out=set()
            for this in self._faces:
                out = out.union(this.level(K))
            return out
        elif self.N==K:
            return set([self])
        return set()


    def bisect(self):
        """Bisect the simplex using longest-edge bisection
    A,B = S.bisect()

This is a persistent method that, for an N-simplex, returns two N-
simplexes resulting from longest-edge bisection.  The order of the 
bisected simplexes is determined by the order of the vertices in the 
longest edge.  "A" will not contain self.longest().V[0].

This algorithm makes certain assumptions regarding the simplex data
tree.  The base of the tree must be Edges, and the faces at each level
must be properly linked to their respective subordinate faces 
(sub-faces).
"""
        if not self._children:
            # If this is an edge, then create two new edges
            if self.N==1:
                a,b = self.V
                c = self.midpoint()
                A = SimplexN(V=(a,c))
                B = SimplexN(V=(c,b))
                # If a length calculation persists, pass that
                # knowledge on to the children
                if self._length:
                    A._length = self._length/2.
                    B._length = A._length
                # Make the bisection persistent
                self._children = A,B
                
            elif self.N>1:
#==============================================================================
# This segment of the code bisects simplexes that are greater than order 1. 
# This is done by identifying the longest edge and finding its midpoint.  The 
# new simplexes are constructed from the original simplex by replacing either
# vertex of the longest edge with the edge's midpoint.  This is the easy part.
#
# The tricky part is properly linking the faces.  When a simplex of order N
# is bisected, the N-1 of the N+1 faces that share the bisected edge will be 
# bisected as well, and their children will become the new faces of the 
# children of the master simplex.  The remaining two faces (those opposite the
# vertices of the bisected edge) do not contain the edge and are not bisected.
# Instead, each will belong to one of the children, and a new order N-1 face 
# will form the boundary between the order N children.
#
# When this method creates a simplex, it must be linked with its subordinate 
# (lower-order) faces.  The 2*(N-1) faces created by bisecting the faces
# adjoining the longest edge are dealt with by recursion, and need not be 
# treated specially.  
#
# Each bisection creates three new simplexes that need to be linked; each 
# child, and the new face they share.  This face will need to have its faces 
# (sub-faces) linked as well.  We could approach the problem by asking a 
# recursion to deal with the new face since it is lower order, but because it
# is created apart from any subordinate biseciton recursion, no other instances
# of the bisect() method will be aware of its existance.  Instead we adopt the 
# convention that the method/function creating a simplex is responsible for
# linking it properly.
#    
# It is reasonable to wonder if this approach would require bisect() to wander
# fully into the recursion depth to link faces of faces of faces.  Fortunately, 
# we do not need to create the faces of the new face, because they will already
# have been created by the recursive bisection of the existing faces.  We only 
# need to find them and link them.  One of them will already exist; it is the
# order N-2 face opposite both of the vertices of the longest edge.
#
# The careful reader may notice that each recursion will individually look
# for the longest edge and calculate its midpoint, causing lots of redundant
# calculations.  This problem is solved by the persistence of the length,
# longest edge, and midpoint properties.  See the longest(), length(), and
# midpoint() methods for how this is managed.
#==============================================================================
                
                # Find the longest edge
                LE = self.longest()
                a,b = LE.V  # Find the end vertices
                ai = self.V.index(a)    # what are their indices in self?
                bi = self.V.index(b)
                # Find the longest edge midpoint vertex
                c = LE.midpoint()
                # Create two new simplexes from the original vertices
                A = SimplexN(V=self.V)
                B = SimplexN(V=self.V)
                
                # Replace the opposing vertex with c
                A.V[bi] = c
                B.V[ai] = c
                # At this point A and B contain the correct vertices
                # Now, we need to deal with the faces.
                
                # create the new face along the bisection
                # The new face will be comprised of all vertices that are not
                # a and b, and the new vertex, c
                newV = list(self.V)
                newV.remove(a)
                newV.remove(b)
                newV.append(c)
                new_face = SimplexN(V=newV)
                
                # Now, we need to bisect all faces sharing the longest edge
                # Loop through the faces
                for index in range(self.N+1):
                    # If the face is opposite a, don't bisect it, link it as is
                    if index == ai:
                        B._faces[index] = self._faces[index]
                    # If the face is opposite b, link it directly
                    elif index == bi:
                        A._faces[index] = self._faces[index]                        
                    # If this face shares the longest edge
                    # It needs to be bisected
                    else:
                        FA,FB = self._faces[index].bisect()
                        # Because the order of A and B is based on the order
                        # of vertices in the longest edge, there is no need
                        # to make sure FA and FB aren't switched.  FA is in A
                        # and FB is in B
                        A._faces[index] = FA
                        B._faces[index] = FB
                        # The bisection of this face will create a new
                        # sub-face that is shared between A, B, and new_face
                        if FA.N>1:
                            # The new sub-face will be located opposite a
                            fai = FA.V.index(a)
                            # The addface method will make sure the indices
                            # are handled correctly
                            new_face._addface(FA._faces[fai])
                
                # Finally, if there are subfaces, link the existing subface 
                # opposite the new vertex.  This is a sub-face opposite both
                # a and b.
                FA = self._faces[ai]
                if FA.N>1:
                    # Find b
                    fbi = FA.V.index(b)
                    # The sub-face opposite b in the face opposite a will
                    # be the sub-face opposite c in the new_face
                    new_face._faces[-1] = FA._faces[fbi]
                
                # add the new face to the children
                # No need to evoke _addface() we already know ai and bi
                A._faces[ai] = new_face
                B._faces[bi] = new_face

                # point the children to their parent
                A._parent = self
                B._parent = self
                # make the result persistent
                self._children = A,B
        
        return self._children


    def longest(self):
        """Return the longest edge (1-simplex) of the simplex"""
        if not self._longest:
            # If there are faces
            if self.N>1:
                # Collect the longest edges of the faces
                edges = set()
                for face in self._faces:
                    edges.add(face.longest())
                self._longest = edges.pop()
                while edges:
                    test = edges.pop()
                    if self._longest.lengthx() < test.lengthx():
                        self._longest = test
            elif self.N==1:
                return self
            else:
                raise Exception("Cannot find the length of a vertex")
        return self._longest



    def areaf(self,P=None):
        """Calculate the area of an N-1 simplex in N space
    A = S.areaf()
        or
    A = S.areaf(P)
"""
        pass
    

    def distf(self,P=None):
        """Calculate the f-space distance between P and the simplex
    d = S.distf()
        or
    d = S.distf( P )

Here, P is a point of any dimension higher than N.  If P is not 
specified, the origin will be used.  P can be any array-like object.

Options A B and C are not intended for commandline use.  They are 
keywords that allow recursive calls to benefit from preliminary 
calculations done by parent simplexes.
"""
        # for convenience
        N = self.N
        # dimension of the space
        M = len(self.V[0].Y)
        # Default to the origin
        if P==None:
            P = np.zeros((M))
        # If P is not the same order as the vertices
        elif len(P)!= M:
            raise Exception(
'Simplex vertices have dimension {:d}, but P only has dimension {:d}.'.format(M,len(P)))

        # Calculate basis vectors
        B = np.zeros((M,N))
        for index in range(N):
            B[:,index] = self.V[index+1].Y - self.V[0].Y
        # Calculate the NxN symmetric solution matrix
        A = np.dot(B.T,B)
        # Record the "origin" for the canonical coordinate system
        Y0 = self.V[0].Y
        # Calculate the vector, C (dim N)
        C = np.dot(P-Y0,B)
        
        # Solve for the projection in canonical coordinates (Z)
        Z = np.linalg.solve(A,C)
        # First, check to see if a Z value is negative
        for index in range(N):
            # If so, then the projected point, Q is outside of the simplex
            if Z[index] < 0:
                if self.N ==1:
                    # If this is just an edge, just grab the end vertex
                    return np.linalg.norm(P-Y0)
                else:
                    # recurse to the opposite face
                    return self._faces[index+1].distf(P=P)
        # If the sum of z values is greater than unity, then Q is outside of
        # the simplex
        if Z.sum()>1.:
            if self.N==1:
                # If this is just an edge, just grab the end vertex
                return np.linalg.norm(P-self.V[1].Y)
            else:
                # recurse to the zero-face
                return self._faces[0].distf(P=P)
        
        # Otherwise, Q is in the current simplex
        # Reconstruct Q from the canonical projection
        Q = np.dot(B,Z.T)+Y0
        return np.linalg.norm(P - Q)





    def midpoint(self):
        """Return the midpoint of a simplex
    vm = S.midpoint()
   
Midpoint is a persistent method.  It returns a vertex at the simplex's
ceter in X-space.
"""
        if not self._midpoint:
            # Create an iterator for the vertex set
            it = iter(self.V)
            # Grab the first element only
            v0 = next(it)
            # pull the function pointer and the first x-value
            X = np.array(v0.X)
            f = v0.f
            for v in it:
                X += v.X
            X /= (self.N+1)
            self._midpoint = Vertex(f,X)
        return self._midpoint
    
    

    def plot3(self, *argv, **kwarg):
        """Produce a 3D representation of the simplex
    S.plot3()
    
Valid only for 3-simplexes, the plot3() method represents the vertices
and edges of a simplex.


"""
        if 'axes' in kwarg:
            axes = kwarg.pop('axes')
        else:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
        edges = self.level(1)
        for ee in edges:
            if ee:
                a,b = ee.V
                axes.plot((a.X[0],b.X[0]),(a.X[1],b.X[1]),(a.X[2],b.X[2]), *argv, **kwarg)
        return axes



#==============================================================================
# Edge
#   Child of SimplexN
# The Edge class is a special instance of the N-Simplex where N==1.  They are
# comprised of two vertices.  The Edge class exposes three unique methods.
# 
# lengthx(), lengthf(), and an overload of bisect()
#==============================================================================


class Edge(SimplexN):
    """Special instance of a Simplex that is order 1
    The Edge is order 1 (self.N == 1), and is comprised of two vertices
This special case permits three special methods:

lengthx(), lengthf(), and an overload of the bisect() method.
"""
    def __init__(self,copy=None,V=None):
        # perform the standard initialization
        SimplexN.__init__(self,copy=copy,V=V)
        if self.N!=1:
            raise Exception('Edges MUST be order 1')
        # The longest edge is always self.  If someone is silly enough to ask,
        # don't bother to look.
        self._longest = self
        # Add persistent length calculations
        self._lengthx = None
        self._lengthf = None
        if isinstance(copy,Edge):
            self._lengthx = copy._lengthx
            self._lengthf = copy._lengthf
        
    
    def lengthx(self):
        """Find the length of an edge in x-space.
    Lx = E.lengthx()
    
Lx is a scalar indicating the geometric length of the line segment
formed between the x-coordinates of the vertices forming the edge.

lengthx is a persistent property.
"""
        if self._lengthx is None:
            va,vb = tuple(self.V)
            self._lengthx = np.linalg.norm(va.X - vb.X)
        return self._lengthx



    def lengthf(self):
        """Approximate the length of an edge in f-space.
    Lf = E.length()
    
Lf is a scalar indicating the geometric length of the line segment 
formed between the f-coordinates of the vertices forming the edge.
Though lengthx() returns the actual edge length, lengthf() is merely an
approximation.  In x-space, the edge is, by definition, a line segment,
but it may be a curve in f-space.

lengthf is a persistent property.
"""
        if self._lengthf is None:
            va,vb = tuple(self.V)
            self._lengthf = np.linalg.norm(va.Y - vb.Y)
        return self._lengthf



    def bisect(self):
        """Bisect the edge
    A,B = E.bisect()

Edge bisection results in two new edges; each containing one of the 
original vertices.  They share the midpoint of the edge as their new
vertex.

The children of bisection are persistent.
"""
        if self._children is None:
            a,b = self.V
            c = self.midpoint()
            A = SimplexN(V=(a,c))
            B = SimplexN(V=(c,b))
            # If a length calculation persists, pass that
            # knowledge on to the children
            if self._length:
                A._length = self._length/2.
                B._length = A._length
            # Make the bisection persistent
            self._children = A,B
        return self._children


class _vertexiter:
    """Iterator class for vertices
>>> for V in _vertexiter(V,NN):
...

This will iterate over every unique combination of verticies that can
form a NN-simplex.
"""
    def  __init__(self,V,NN):
        self.V = V
        self.K = NN+1
        self.N = len(V)
        # initialize the index list
        self.II = [NN]
        while self.II[-1]>0:
            self.II.append(self.II[-1]-1)
        self.II[0]-=1

    def __iter__(self):
        return self

    def next(self):
        self.II = self.increment(self.II,self.N)
        return [self.V[ii] for ii in self.II]

    def increment(self, II, K ):
        """Recursive incrementing function"""
        II[0] += 1
        if II[0]==K:
            # We're at the root and we've wrapped
            if len(II)==1:
                raise StopIteration()
            II[1:] = self.increment(II[1:], K-1)
            II[0] = II[1]+1
        return II     
        
            


def simplex(V):
    """Form a fully populated simplex from base vertices"""
    V = list(V)
    N = len(V)-1  # number of vertices

    n1list = []
    # Loop through the simplex levels
    # Stop at N
    for kk in range(1,N+1):
        n2list = n1list  # The n2list represents the last level list of faces
        n1list = []      # the n1list is the current level list of faces
        # Loop through all appropriate combinations of vertices
        for VV in _vertexiter(V,kk):
            # Create a simplex for each
            new = SimplexN(V=VV)
            # Link the faces from the last level
            for this in n2list:
                new._addface(this)
            # Add this face to the current level
            n1list.append(new)
    # On the last iteration, when kk==N, _vertexiter will only allow the
    # for loop to execute once; creating a new simplex containing all vertices
    # That is the new master simplex.
    return new
    



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