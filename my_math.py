#!/usr/bin/env python
# coding: utf-8

# In[5]:


import math
import copy
from inspect import isfunction


# <h3>Primes</h3>

# In[6]:


def primality_brute(N):
    """(int) -> (boolean)
    Does a brute force test of primailty by checking the divisibility of N by all integers less than sqrt(N).
    """
    if N == 1: #1 is not a prime by definition
        return False
    
    if type(N) != int: #nonintegers cannot be prime
        raise TypeError("N must be an integer.")
    
    stop = math.sqrt(N) #stop point
    i = 2
    while i <= stop:
        if N % i == 0: #if i divides N
            return False
        i += 1
    return True



def primes_less_than(N):
    """ (int) -> (list)
    Returns a list of all prime numbers strictly less than N.
    
    >>> primes_less_than(4)
    [2, 3]
    
    >>> primes_less_than(10)
    [2, 3, 5, 7]
    
    >>> primes_less_than(1000)
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 
    557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 
    661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 
    809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 
    937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

    """
    primes = [i for i in range(2, N)]
    
    for x in primes: #iterates through primes
        multiples = 2
        while multiples*x < N: #removes all multiples of x from list of primes
            if multiples*x in primes:
                primes.remove(multiples*x)
            multiples += 1
    
    return primes


# In[7]:


### Code for implementing the method of repeated squares in determining a^m % n

def powers_of_two(num):
    """ (int) -> (list)
    Returns a list containing the exponents to which we raise each consecutive term in our sum of powers of two representation of num.
    
    >>> powers_of_two(6)
    [1, 2]
    
    >>> powers_of_two(57)
    [0, 3, 4, 5]
    """
    num_bin = bin(num)[2:] #converts num to a binary string and remvoes '0b' at the front
    
    exponents = []
    for pos in range(len(num_bin)): #iterates backwards through the string
        if num_bin[len(num_bin) - 1 - pos] != '0':
            exponents.append(pos)
    
    return exponents


def helper_modulo(base, exponent, modulo):
    """ (int, int, int) -> (int)
    Returns (base^(2^exponent)) (mod modulo)
    
    >>> helper_modulo(271, 2, 481)
    16
    
    >>> helper_modulo(271, 6, 481)
    419
    
    >>> helper_modulo(4, 3, 3)
    1
    """
    if exponent == 0: #base case --> x^(2^0)=x for any x
        return base % modulo
    else:
        return ((helper_modulo(base, exponent-1, modulo))**2) % modulo # (a^(2^x))^2 = (a^(2^(x+1))) (mod n)
    


def repeated_squares(base, power, modulo):
    """ (int, int) -> (int)
    Calculates base^(power) (mod modulo) using the method of repeated squares.
    
    >>> repeated_squares(271, 321, 481)
    47
    
    >>> repeated_squares(50921, 30, 5)
    1
    """
    power_bin = powers_of_two(power) #finds the power as a sum of powers of twos
    
    answer = 1 #Beginning of answer
    
    for elements in power_bin: #iterates through the factors that make up base^power
        answer *= helper_modulo(base, int(elements), modulo)    
    
    return answer % modulo


# In[8]:


##### Implementation of Selfridge's Conjecture of Prime Numbers #####

def fibonacci(k):
    """ (int) -> (int)
    Returns the kth Fibonacci number, F0=0, F1=1, Fn=Fn-1 +Fn-2 for n > 1
    
    >>> fibonacci(0)
    0
    
    >>> fibonacci(1)
    1
    
    >>> fibonacci(8)
    21
    """
    if type(k) != int: #input validation
        raise TypeError("This function requires and integer input.")
    
    if k == 0 or k == 1: #F0=0 F1=1
        return k
    else: #Fn=Fn-1 +Fn-2
        return fibonacci(k-1) + fibonacci(k-2)


def selfridge(N):
    """ (int) -> (boolean)
    Uses Selfridge's conjecutre to test N for primality. This is not a conclusive test since the conjecture
    has not yet been proven.
    If N is odd and N % 5 = +-2 the N is prime if:
    2**(N-1)%N=1
    and
    F(N+1)%N=0
    
    >>> selfridge(17)
    True
    
    >>> selfridge(13)
    True
    
    >>> selfridge(2)
    False
    
    >>> selfridge(0)
    False
    """
    if type(N) != int:
        raise TypeError("Only integers can be primes.")
    
    if N%2 == 0:
        return False
    if N%5 not in [2, 3]:
        raise ValueError("Selfridge's conjecture does not apply to this number.")
    
    if repeated_squares(2, N-1, N) != 1:
        raise ValueError("Selfridge's conjecture fails to apply to this number. Inconclusive test.")
    
    if fibonacci(N+1)%N != 0:
        raise ValueError("Selfridge's conjecture fails to apply to this number. Inconclusive test.")
    
    return True


# In[9]:


def factor(N):
    """ (int) -> (list)
    Returns a list of the prime factors of N.
    """
    factors = []
    if N < 0:
        factors.append(-1)

    if type(N) is not int:
        raise TypeError("Function only factors integers.")
    
    if N==1 or primality_brute(N): #special cases
        return [N]

    #first make a list of all possible prime factors of N
    possible_factors = primes_less_than(N)

    current = N
    while current > 1:
        for entry in possible_factors:
            if current%entry == 0:
                factors.append(entry)
                current = current//entry #divide current by its factor entry to get our next number to factor
                continue
            possible_factors.remove(entry) #if current is not divisible by an entry in possible_factors we can remove that entry completely
    return factors


# <h3>Vectors</h3>

# In[31]:


class Vector:
    """
    A class to represent a row Vector and define Vector algebra and operations.

    Attributes
    ------
    data : list
        Contains the entries of the vectors.
    
    Methods
    ------
    angle(v2):
        Angle between two Vectors.
    antiparallel(v2):
        Returns True if the instance Vector and v2 Vector are anti-parallel. False otherwise.
    cross_prod(v2):
        Cross product of instance Vector and v2; only defined in 3D.
    cylindrical():
        Prints a 3D Vector's cylindrical coordinates.
    dot(v2):
        Dot product of two Vectors.
    magnitude():
        Finds the magnitude of the Vector.
    opposite(v2):
        Returns True if the instance Vector and v2 Vector are opposite. False otherwise.
    parallel(v2):
        Returns True if the instance Vector and v2 Vector are parallel. False otherwise.
    perpendicular(v2):
        Returns True if the instance Vector and v2 Vector are perpendicular. False otherwise.
    scalar_proj(v2):
        Scalar projection of the instance Vector onto the v2 Vector.
    spherical():
        Prints a 3D Vector's spherical coordinates.
    to_matrix():
        Converts the instance Vector to a one row Matrix.
    transpose():
        Transposes the instance Vector from a row Vector to a column Vector. This is represented in Matrix form.
    unit():
        Returns the unit Vector in the direction of the instance Vector.
    vector_proj(v2):
        Vector projection of the instance Vector onto the v2 Vector.

    """

    def __init__(self, data=None):
        """ (self, list) -> (Vector)
        Overloading constructor method for defining instances of the Vector class. Will create a Vector with entries and size given by data.
        """
        if data is None: #empty Vector
            self.data = []
        elif not (type(data) is list):
            raise TypeError("A vector must be initilaized with a numeric list.")
        else:
            for entry in data:
                if type(entry) not in [float, int]:
                    raise TypeError("Entries of a vector must be numeric.")
            self.data = data
    
    def __len__(self):
        """ (self) -> (int)
        Returns the dimensionality of the self Vector.
        """
        return len(self.data)
    
    def __getitem__(self, key):
        """ (self, int) -> (num)
        Indexing into the self Vector.
        """
        return self.data[key]
    
    
    def __setitem__(self, key, value):
        """ (self, int, num) -> ()
        Reset self Vector's key coordinate to value.
        """
        self.data[key] = value
    
    
    def __contains__(self, value):
        """ (self, num) -> (boolean)
        Checks if value is one of the components of the self Vector.
        """
        return value in self.data
    
    def __iter__(self):
        """ (self) -> (Vector)
        Returns an iterator.
        """
        self.n = 0
        return self
    
    def __next__(self):
        """ (self) -> (num)
        Goes to the next object in the iterator.
        """
        if self.n < len(self):
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration
            
    def __str__(self):
        """ (self) -> (str)
        Returns a meaningful string representation of the self Vector. 
        """
        if len(self) == 0:
            return "<>"
        return "<" + str(self.data)[1:-1] + ">" 
    
    def __eq__(self, v2):
        """ (self, Vector) -> (boolean)
        Vectors are equal if they are equal component-wise.
        """
        return self.data == v2.data

    def __ne__(self, v2):
        """ (self, Vector) -> (boolean)
        Vectors are not equal if they are not equal component-wise.
        """
        return not self==v2
        
    
    def __mul__(self, a):
        """ (self, num/Matrix) -> (Vector/Matrix)
        Defines multiplication of a Vector by a scalar or a Matrix. Returns another Vector or a Matrix depending on the nature of the operation.
        """
        if type(a) is Matrix:
            if len(self) != a.count_rows():
                raise ValueError("The Vector must have the same number of elements as the Matrix has rows.")
            return Matrix([self.data])*a
                    
        if type(a) not in [float, int]:
            raise TypeError("A vector can only by multiplied by a scalar or a Matrix.")
        
        new_data = []
        for i in self:
            new_data.append(a*i)
        return Vector(new_data)
    
    def __rmul__(self, a):
        """ (self, num) -> (Vector)
        Defines multiplication of a Vector by a scalar. Returns another Vector.
        """
        if type(a) not in [float, int]:
            raise TypeError("A vector can only by multiplied by a scalar or a Matrix.")
            
        new_data = []
        for i in self:
            new_data.append(a*i)
        return Vector(new_data)
    
    def __truediv__(self, a):
        """ (self, num) -> (Vector)
        Defines division of a Vector by a scalar. Returns another Vector.
        """
        if type(a) not in [float, int]:
            raise TypeError("A vector can only by divided by a scalar.")
            
        new_data = []
        for i in self:
            new_data.append(i/a)
        return Vector(new_data)
    

    def __add__(self, v2):
        """ (self, Vector) -> (Vector)
        Defines addition of a Vector by another Vector.
        """
        if type(v2) != Vector:
            raise TypeError("Vectors can only added to other vectors.")
        if len(self) != len(v2):
            raise ValueError("We can only add vectors with the same lengths.")
            
        new_data = []
        for i in range(len(self)):
            new_data.append(self[i] + v2[i])
        return Vector(new_data)
    

    def __sub__(self, v2):
        """ (self, Vector) -> (Vector)
        Defines subtractiong of a Vector by another Vector.
        """
        return self + (-1*v2)
        
    def magnitude(self):
        """ (self) -> (num)
        Finds the magnitude of the self Vector.
        """
        mag = 0
        for i in self:
            mag += i**2
        return math.sqrt(mag)
    
    def dot(self, v2):
        """ (self, Vector) -> (num)
        Dot product of two Vectors.
        """
        if len(self) != len(v2):
            raise ValueError("Both vectors must have the same length.")
        count = 0
        for i in range(len(self)):
            count += self[i]*v2[i]
        return count
    
    
    def angle(self, v2):
        """ (self, Vector) -> (num)
        Angle between two Vectors.
        """
        s_mag = self.magnitude()
        v2_mag = v2.magnitude()
        dot_prod = self.dot(v2)
        return math.acos(dot_prod/v2_mag/s_mag)
    

    def scalar_proj(self, v2):
        """ (self, Vector) -> (num)
        Scalar projection of the self Vector onto the v2 Vector.
        """
        return self.dot(v2)/v2.magnitude()
    

    def vector_proj(self, v2):
        """ (self, Vector) -> (Vector)
        Vector projection of the self Vector onto the v2 Vector.
        """
        return self.scalar_proj(v2)*v2.unit()
    
    
    def cross_prod(self, v2):
        """ (self, Vector) -> (Vector)
        Cross product of self and v2 Vectors; only defined in 3D.
        """
        if len(self) != len(v2):
            raise ValueError("Both vectors must have the same length.")
        elif len(self) == 3:
            x = self[2]*v2[3]-self[3]*v2[2]
            y = self[3]*v2[1]-self[1]*v2[3]
            z = self[1]*v2[2]-self[2]*v2[1]
            return Vector([x, y, z])
        raise ValueError("Vectors must both be of length 3.")
        
    
    def spherical(self):
        """ (self) -> ()
        Prints a 3D Vector's spherical coordinates.
        """
        if len(self) != 3:
            raise ValueError("Spherical coordinates are only defined in 3 dimensions.")
        r = self.magnitude() 
        theta = math.atan(self[1]/self[0])
        phi = math.acos(self[2]/r)
        print(r, "r +", theta, "theta +", phi, "phi")
    

    def cylindrical(self):
        """ (self) -> ()
        Prints a 3D Vector's cylindrical coordinates.
        """
        if len(self) != 3:
            raise ValueError("Cylindrical coordinates are only defined in 3 dimensions.")
        r = math.sqrt(self[0]**2 + self[1]**2)
        theta = math.atan(self[1]/self[0])
        z = self[2]
        print(r, 'r +', theta, 'theta +', z, "z")
    

    def unit(self):
        """ (self) -> (Vector)
        Returns the unit Vector in the direction of the self Vector.
        """
        return self/self.magnitude()
    
    def parallel(self, v2):
        """ (self, Vector) -> (boolean)
        Returns True if the self Vector and v2 Vector are parallel. False otherwise.
        """
        return self.dot(v2) == self.magnitude()*v2.magnitude()
    
    def antiparallel(self, v2):
        """ (self, Vector) -> (boolean)
        Returns True if the self Vector and v2 Vector are anti-parallel. False otherwise.
        """
        return self.dot(v2) == -self.magnitude()*v2.magnitude()
    

    def opposite(self, v2):
        """ (self, Vector) -> (boolean)
        Returns True if the self Vector and v2 Vector are opposite. False otherwise.
        """
        return self.antiparallel(v2) and (self.magnitude() == v2.magnitude())
    

    def perpendicualr(self, v2):
        """ (self, Vector) -> (boolean)
        Returns True if the self Vector and v2 Vector are perpendicular. False otherwise.
        """
        return self.dot(v2) == 0
    
    def transpose(self):
        """ (self) -> (Matrix)
        Transposes the self Vector from a row vector to a column vector represented as a Matrix.
        """
        new_data = []
        for entry in self.data:
            new_data.append([entry])
        return Matrix(new_data)
    
    def to_matrix(self):
        """ (self) -> (Matrix)
        Converts the self Vector to a 1xlen(self) Matrix.
        """
        return Matrix([(self.data)])


# <h3>Matrices</h3> <!-- Matrix Link -->

# In[60]:


class Matrix:
    """
    A class to represent a Matrix and define Matrix algebra and operations.

    Attributes
    ------
    data : 2D list
        Contains the entries of the Matrix.
    
    Methods
    ------
    adjugate():
        Returns the adjugate of a square Matrix.
    count_cols():
        Counts the number of columns in the Matrix.
    count_rows():
        Counts the number of rows in the Matrix.
    determinant():
        Returns the determinant of a square Matrix.
    getcols():
        Returns the columns of a Matrix.
    getrows():
        Returns the rows of a Matrix.
    identity(n):
        Returns the nxn identity Matrix.
    inverse():
        Returns the inverse of a square Matrix.
    isdiagonal():
        Checks if a Matrix is diagonal.
    isidentity():
        Checks if a Matrix is the identity.
    isinvertible():
        Checks if a Matrix is invertible.
    issquare():
        Checks if a Matrix is square.
    minor(i, j):
        Returns the i,j minor of the instance Matrix.
    plus_minus_ones(n,m=None):
        Returns an nxm Matrix of alternating +-1.
    remove_col(j):
        Returns a copy of the instance Matrix without the jth column.
    remove_row(i):
        Returns a copy of the instance Matrix without the ith row.
    solve_linear(b,vec=False):
        Solves the linear system A*x=b for x where A is the instance Matrix. b can be a Vector, list, tuple, row Matrix, or a column Matrix.
    to_vector():
        Converts a row or a column Matrix to a row Vector.
    trace():
        Returns the trace of a square Matrix.
    transpose():
        Returns the transpose of a Matrix.
    zero(n,m=None):
        Returns an nxm Matrix of 0.
    
    """

    def __init__(self, data=None):
        """ (self, 2D list) -> (Matrix)
        Overloading constructor method for defining instances of the Matrix class. Will create a Matrix with entries and size given by data.
        """
        if data is None:
            self.data = [[]]
        else:
            num_col = len(data[0])
            for row in data:
                if len(row) != num_col:
                    raise ValueError("All rows must have the same length.")
            self.data = data
    
    def __len__(self):
        """ (self) -> (int)
        Returns the total number of entries in the Matrix self.
        """
        return len(self.data)*len(self.data[0]) #rows*columns
    
    def __getitem__(self, row_idx):
        """ (self, int) -> (list)
        Returns the row given by row_idx when indexing through self.data, which is a normal Python list.
        """
        return self.data[row_idx]
    
    def __contains__(self, value):
        """ (self, num) -> (boolean)
        Returns True if value is one of the entries of the Matrix. False otherwise.
        """
        for row in self:
            if value in row:
                return True
        return False
    
    def count_rows(self):
        """ (self) -> (int)
        Counts the number of rows in the Matrix self.
        """
        if self.data == [[]]:
            return 0
        return len(self.data)
    
    def count_cols(self):
        """ (self) -> (int)
        Counts the number of columns in the Matrix self.
        """
        if self.data == [[]]:
            return 0
        return len(self.data[0])

    def __iter__(self):
        """ (self) -> (Matrix)
        Creates an iterator.
        """
        self.n = 0
        return self
    
    def __next__(self):
        """ (self) -> (list)
        Goes to the next object in the iterator.
        """
        if self.n < self.count_rows():
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration
    
    
    def __str__(self):
        """ (self) -> (str)
        Returns meaningful string representation of the self Matrix.
        """ 
        s = ""
        for row in self.data:
            s += str(row) + "\n"
        return s[:-1]
    

    def __eq__(self, M2):
        """ (self, Matrix) -> (boolean)
        Two Matrices are equal if they are equal component-wise.
        """
        return self.data == M2.data
    def __ne__(self, M2):
        """ (self, Matrix) -> (boolean)
        Two Matrices are not equal if they are not equal component-wise.
        """
        return not self == M2
    
    def __mul__(self, M2):
        """ (self, Matrix/Vector) -> (Matrix)
        Defines Matrix multiplication and multiplication of Matrices by Vectors.
        """
        if not type(M2) in [Matrix, Vector]:
            raise TypeError('Matrices can only be multiplied by other Matrices, Vectors, and left multiplied by scalars')

        if M2 is Vector: #converts M2 vector to a 1xlen(M2) Matrix
            M2 = M2.to_matrix()

        if self.count_cols() != M2.count_rows():
            raise ValueError("Matrix dimensions are incompatible.")
        
        m = self.count_rows()
        n = self.count_cols()
        p = M2.count_cols()

        new_data = Matrix.zero(m, p) #mxp matrix of zeroes
        for row_idx in range(m):
            for col_idx in range(p):
                new_data[row_idx][col_idx] = sum([self[row_idx][i]*M2[i][col_idx] for i in range(n)])
        return Matrix(new_data)

    def __rmul__(self, a):
        """ (self, num) -> (Matrix)
        Defines left multiplication of a Matrix by a scalar.
        """
        if type(a) in [int, float]:
            new_data = copy.deepcopy(self.data)
            for row_idx in range(len(new_data)):
                for col_idx in range(len(new_data[0])):
                    new_data[row_idx][col_idx] *= a
            return Matrix(new_data)
        raise TypeError('Matrices can only be multiplied by other Matrices, Vectors, and left multiplied by scalars and Vectors.')

    def __add__(self, M2):
        """ (self, Matrix) -> (Matrix)
        Defines addition of a Matrix to another Matrix.
        """
        num_rows = self.count_rows()
        num_cols = self.count_cols()

        if not type(M2) is Matrix:
            raise TypeError("Can only add matrices to other matrices.")
        if num_cols != M2.count_cols() or num_rows != M2.count_rows():
            raise ValueError("Can only add matrices of the same size.")
        
        new_matrix = Matrix.zero(num_rows, num_cols)
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                new_matrix[row_idx][col_idx] = self[row_idx][col_idx] + M2[row_idx][col_idx]
        return new_matrix

    def __sub__(self, M2):
        """ (self, Matrix) -> (Matrix)
        Defines subtraction of a Matrix to another Matrix.
        """
        num_rows = self.count_rows()
        num_cols = self.count_cols()

        if not type(M2) is Matrix:
            raise TypeError("Can only subtract matrices to other matrices.")
        if num_cols != M2.count_cols() or num_rows != M2.count_rows():
            raise ValueError("Can only subtract matrices of the same size.")
        
        new_matrix = Matrix.zero(num_rows, num_cols)
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                new_matrix[row_idx][col_idx] = self[row_idx][col_idx] - M2[row_idx][col_idx]
        return new_matrix
    
    
    def __truediv__(self, M2):
        """ (self, Matrix) -> (Matrix)
        Matrix division is defined as A/B=A*B**-1.
        """
        if not type(M2) is Matrix:
            raise TypeError("Matrices can only be divided by other matrices.")
        if not M2.isinvertible():
            raise ValueError("Can only divide by invertible matrices.")
        return self*M2.inverse()
    
    def __pow__(self, a):
        """ (self, int) -> (Matrix)
        Returns the square Matrix self exponentiated to the ath power.
        """
        if not type(a) is int:
            raise TypeError("Matrices can only be raised to integer powers.")
        if not self.issquare():
            raise ValueError("Only square Matrices can be exponentiated.")

        if a == 0: #base case for recursion
            return Matrix.identity(self.count_rows())
        if a > 0: #if a is positive it goes down to zero
            return self*(self**(a-1))
        #otherwise a is negative and it goes up to zero
        return self.inverse()*(self.inverse()**(a+1))

    def remove_row(self, i):
        """ (self, int) -> (Matrix)
        Returns a copy of the Matrix self without the ith row. Indexing starts at 0.
        """
        new_data = copy.deepcopy(self.data)
        new_data.pop(i)
        return Matrix(new_data)

    def remove_col(self, j):
        """ (self, int) -> (Matrix)
        Returns a copy of the Matrix self without the jth column. Indexing starts at 0.
        """
        new_data = copy.deepcopy(self.data)
        for row in new_data:
            row.pop(j)
        return Matrix(new_data)
    
    def minor(self, i, j):
        """ (self, int, int) -> (Matrix)
        Returns the i,j minor of Matrix self. Indexing starts at 0.
        """
        new_data = copy.deepcopy(self.data)
        new_data.pop(i)
        for row in new_data:
            row.pop(j)
        return Matrix(new_data)


    
    def determinant(self):
        """ (self) -> (num)
        Returns the determinant of the square Matrix self.
        """
        if not self.issquare():
            raise ValueError("We can only find the determinant of a square matrix.")

        if len(self) == 1:
            return self[0][0]
        if len(self) == 2:
            return self[0][0]*self[1][1] - self[0][1]*self[1][0]
        #recursively find determinant of i,j minor times (-1 or 1)
        tot = 0
        n = self.count_rows() #n is size of matrix

        for i in range(n):
            tot += (-1)**(i+1)*self[i][1]*self.minor(i, 1).determinant()
        return tot
    

    def trace(self):
        """ (self) -> (num)
        Returns the trace of the square Matrix self.
        """
        if not self.issquare():
            raise ValueError("We can only take the trace of a square Matrix.")

        n = self.count_rows() #size of the matrix
        tr = 0 #value we will return
        for i in range(n): # add each element along the main diagonal to tr
            tr += self[i][i]
        return tr


    def adjugate(self):
        """ (self) -> (Matrix)
        Returns the adjugate Matrix of the nxn Matrix self.
        """
        n = self.count_rows()

        adj = Matrix.zero(n)

        for i in range(n):
            for j in range(n):
                adj[i][j] = (-1)**(i+j)*(self.minor(j,i).determinant())
        return adj


    def inverse(self):
        """ (self) -> (Matrix)
        Returns the inverse of the Matrix self if it is invertible.
        """
        detA = self.determinant()
        if detA == 0:
            raise ValueError("This Matrix is not invertible.")
        return (1/detA)*self.adjugate()

    
    def to_vector(self):
        """ (self) -> (Vector)
        Converts a 1xn Matrix to a row Vector. If self is a mx1 column Vector it will also be converted to a row Vector.
        """
        if self.count_rows() == 1:
            return Vector(self.data[0])

        if self.count_cols() == 1:
            return Vector(self.getcols()[0])

        raise ValueError("Matrix dimensions are incompatible with vectorization.")


    def solve_linear(self, b, vec=False):
        """ (self, Vector/list/tuple/Matrix, boolean) -> (Matrix/Vector)
        Solves the linear system self*x=b for x. Returns x as a Matrix if vec is False, returns x as a row Vector if vec is True.
        This method is flexible with the form of b, it will accept a row Vector, list, tuple, or an nx1/1xn Matrix so long as the dimensions work. 
        Currently cannot solve systems where self is not invertible. 
        """
        if type(b) is Vector:
            b = b.data
        elif type(b) in [tuple, list]:
            b = list(b)
        elif type(b) is Matrix:
            if b.count_cols() == 1:
                b = b.get_cols()[0]
            elif b.count_rows() == 1:
                b = b.get_rows()[0]
            else:
                raise ValueError("Matrix dimensions incompatible with solving linear system.")
            
            
        
        #A*x=b
        detA = self.determinant()
        if detA == 0:
            raise ValueError("This method cannot currently solve linear systems for non-invertible Matrices.")
        
        x = [] #solution to system
        n = len(b) #size of system
        for i in range(n):
            A_ith = self.transpose().data
            A_ith[i] = b
            x.append(Matrix(A_ith).determinant()/detA)
        
        if vec: #parameter to choose whether or not we want to return a Vector of a Matrix
            return Vector(x)
        else:
            final_data = []
            for entry in x:
                final_data.append([entry])
            return Matrix(final_data)



    def getrows(self):
        """ (self) -> (list)
        Returns a list of all the rows of self.
        """
        return self.data
    

    def getcols(self):
        """ (self) -> (list)
        Returns a list of all the columns of self.
        """
        num_row = self.count_rows()
        num_col = self.count_cols()
        return [[self[i][j] for i in range(num_row)] for j in range(num_col)]


    def transpose(self):
        """ (self) -> (Matrix)
        Returns the transpose of the Matrix self.
        """
        new = Matrix()
        new.data = self.getcols()
        return new
        


    def issquare(self):
        """ (self) -> (boolean)
        Returns True if the self Matrix is square. False otherwise.
        """
        return self.count_cols()==self.count_rows()
    

    def isdiagonal(self):
        """ (self) -> (boolean)
        Returns True if the self Matrix is diagonal. False otherwise.
        """
        num_row = self.count_rows()
        num_col = self.count_cols()
        
        if num_row != num_col:
            return False

        for i in range(num_row):
            for j in range(num_col):
                if i!=j and self[i][j]!=0: #if entries off of main diagonal are not identically 0
                    return False
        return True
    

    def isidentity(self):
        """ (self) -> (boolean)
        Returns True if the self Matrix is the identity Matrix. False otherwise.
        """
        if not self.isdiagonal(): #identity is diagonal
            return False
        
        num_row = self.count_rows()
        for i in range(num_row):
            if self[i][i] != 1: #if main diagonal is not identically 1 
                return False
        return True


    def isinvertible(self):
        """ (self) -> (boolean)
        Returns True if the self Matrix is invertible. False otherwise.
        """
        return self.determinant() != 0
    

    
    @staticmethod
    def plus_minus_ones(n, m=None):
        """ (int, int) -> (Matrix)
        Returns an nxm Matrix of alternating +-1.
        """
        if m is None: #single input defines square matrix
            m = n
        new_data = [[(-1)**(i+j) for i in range(m)] for j in range(n)]
        return Matrix(new_data)


    @staticmethod
    def zero(n, m=None):
        """ (int, int) -> (Matrix)
        Returns an nxm Matrix of 0.
        """
        if m is None: #single input defines square matrix
            m = n
        new_data = [[0 for i in range(m)] for j in range(n)]
        return Matrix(new_data)


    @staticmethod
    def identity(n):
        """ (int) -> (Matrix)
        Returns an nxn identity Matrix.
        """
        new_matrix = Matrix.zero(n)
        
        i = 0
        while i < n:
            new_matrix[i][i] = 1
            i += 1
        
        return new_matrix


# <h3>Calculus</h3>

# In[1]:


def derivative(func, x, error=10**-5):
    """ (function, float) -> (float)
    Estimates the derivative of func at x with an given error.
    """
    h = math.sqrt(error)
    return (func(x+h)-func(x-h))/(2*h)


# In[ ]:


def list_derivative(x, data):
    """ (list, list) -> (list)
    Estimates the derivative from a list of discrete data points.
    """
    if len(data) in [1, 2]:
        raise ValueError("Cannot estimate the derivative from a small dataset of length 1 or 2.")

    deriv = []

    deriv.append((data[1]-data[0])/(x[1]-x[0])) #endpoints

    for i in range(1, len(data)-1):
        deriv.append((data[i+1]-data[i-1])/(x[i+1]-x[i-1]))

    deriv.append((data[-1]-data[-2])/(x[-1]-x[-2])) #endpoints

    return deriv


# In[9]:


def riemann_integral(func, a, b, bins=500, side='mid'):
    """ (function, num, num, int, str) -> (num)
    Returns an estimate for the integral of func from a to b, a<=b. The estimate is determined using
    Riemann sums. The side parameter determines the whether the bins should be right-sided, midpoint, 
    or left-sided; default is midpoint sum."""
    if b < a:
        return ValueError('The left limit must be less than or equal to the right limit.')
    if a == b:
        return 0
    
    step = (b-a)/bins #width of each bin
    
    total = 0 #value of the estimate
    current = a #start at left endpoint
    if side == 'right':
        while (current+step) <= b:
            total += step*func(current+step)
            current += step
    elif side == 'mid':
        while current < b:
            total += step*func(current+step/2)
            current += step
    elif side == 'left':
        while current <= b:
            total += step*func(current)
            current += step
    else:
        return ValueError("side parameter must be right, mid, or left.")
    
    return total


def trapezoid_integral(func, a, b, steps=100):
    """ (function, num, num, int) -> (num)
    Returns an estimate for the integral of func from a to b, a<=b. The estimate is determined using
    the trapezoid rule. The formula for one step is given by
        0.5*(xn+1 - xn)*(func(xn+1) + func(xn))
    """
    if b < a:
        return ValueError('The left limit must be less than or equal to the right limit.')
    if a == b:
        return 0
    
    stepsize = (b-a)/steps #width of each step
    
    total = 0 #value of the estimate
    current = a + stepsize #start at left endpoint
    while current <= b:
        total += 0.5*stepsize*(func(current) + func(current-stepsize))
        current += stepsize
    
    return total


# In[10]:


def euler_odes(func, times, y0):
    """ (func, list, num) -> (list)
    Estimates the numerical solution to the ODE y'(t)=func(y, t) with initial value y0 at times[0].
    times is the list of times where we want to approximate our solution. Returns a list of our approximations
    of y at each of the points in times."""
    y = [0]*len(times)
    y[0] = y0
    for i in range(len(times)-1):
        y[i+1] = y[i] + func(y[i], times[i])*(times[i+1]-times[i])
    return y


# <h3>Vector Functions</h3>

# In[64]:


class VectorFunction:
    """
    A class to represent single-parameter vector functions and define the algebra and operations on said functions.

    Attributes
    ------
    data : list
        Contains the functions that define a VectorFunction.
    
    Methods
    ------
    angle(x,vec):
        Finds the angle between the instance VectorFunction at x and the given Vector.
    antiparallel(x,vec):
        Returns True if the instance VectorFunction evaluated at x is anti-parallel to the vec Vector. False otherwise.
    arc_length(start,stop,steps=100):
        Approximates the arc length of the instance VectorFunction between start and stop.
    binormal(x,error=10**-5):
        Approximates the binormal Vector of the instance VectorFunction at x. Only defined in 3D.
    eval_derivative(x,error=10**-5):
        Approximates the derivative of the instance VectorFunction at x.
    magnitude():
        Returns a function that finds the magnitude of the instance VectorFunction when called.
    normal(x,error=10**-5):
        Approximates the unit normal Vector of the instance VectorFunction at x.
    opposite(x,vec):
        Returns True if the instance VectorFunction evaluated at x is opposite to the vec Vector. False otherwise.
    parallel(x,vec):
        Returns True if the instance VectorFunction evaluated at x is parallel to the vec Vector. False otherwise.
    perpendicular(x,vec):
        Returns True if the instance VectorFunction evaluated at x is perpendicular to the vec Vector. False otherwise.
    scalar_proj(x,vec):
        Finds the scalar projection of the instance VectorFunction at x onto the given Vector.
    unit(x):
        Returns the unit Vector in the direction of the instance VectorFunction evaluated at x.
    vector_proj(x,vec):
        Finds the vector projection of the instance VectorFunction at x onto the given Vector.
    tangent(x,error=10**-5):
        Approximates the unit tangent Vector of the instance VectorFunction at x.
    """

    def __init__(self, data=None):
        """ (self, list) -> (VectorFunction)
        Overloading constructor method for defining instances of the VectorFunction class. Will create a VectorFunction with entries and size given by data.
        """
        if data is None:
            self.data = []
        elif not (type(data) is list):
            raise TypeError("A VectorFunction must be initilaized with a list of single-variable functions.")
        else:
            for entry in data:
                if not isfunction(entry):
                    raise TypeError("Entries of a VectorFunction must be single-variable functions.")
            self.data = data


    def __call__(self, t):
        """ (self, num) -> (Vector)
        Evaluates a VectorFunction at t, returns the result as a Vector.
        """
        return Vector([func(t) for func in self])

    
    def __len__(self):
        """ (self) -> (int)
        Returns the dimensionality of the VectorFunction.
        """
        return len(self.data)
    
    
    def __getitem__(self, key):
        """ (self, int) -> (func)
        Allows us to index into a VectorFunction.
        """
        return self.data[key]
    

    def __setitem__(self, key, value):
        """ (self, int, func) -> ()
        Allows us to set the value of the VectorFunction at the key index.
        """
        if not isfunction(value):
            raise TypeError("VectorFunctions only take functions as entries.")
        self.data[key] = value
    

    def __contains__(self, func):
        """ (self, func) -> (boolean)
        Checks if the function value is one of the entries of the self VectorFunction.
        """
        return func in self.data
    

    def __iter__(self):
        """ (self) -> (VectorFunction)
        Creates an iterator object.
        """
        self.n = 0
        return self
    
    def __next__(self):
        """ (self) -> (func)
        Goes to the next object in the iterator.
        """
        if self.n < len(self):
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration
            

    def __str__(self):
        """ (self) -> (str)
        Returns a string representation of the self VectorFunction
        """
        if len(self) == 0:
            return "<>"

        return "<" + str(self.data)[1:-1] + ">"
    
    
    def __eq__(self, v2):
        """ (self, VectorFunction) -> (boolean)
        VectorFunctions are equal if they are equal component-wise.
        """
        return self.data == v2.data
    def __ne__(self, v2):
        """ (self, VectorFunction) -> (boolean)
        VectorFunctions are not equal if they are not equal component-wise.
        """
        return not self==v2
    

    def __mul__(self, a):
        """ (self, num) -> (VectorFunction)
        Defines scalar multiplication of a VectorFunction.
        """
        if type(a) not in [float, int]:
            raise TypeError("A VectorFunction can only be multiplied by a scalar.")
        
        new_data = []
        def make_new(i):
            def temp(x):
                return a*self[i](x)
            return temp

        for i in range(len(self)):
            new_data.append(make_new(i))
        return VectorFunction(new_data)
    

    def __rmul__(self, a):
        """ (self, num) -> (VectorFunction)
        Defines scalar multiplication of a VectorFunction.
        """
        if type(a) not in [float, int]:
            raise TypeError("A VectorFunction can only be multiplied by a scalar.")
        
        new_data = []
        def make_new(i):
            def temp(x):
                return a*self[i](x)
            return temp

        for i in range(len(self)):
            new_data.append(make_new(i))
        return VectorFunction(new_data)
    

    def __truediv__(self, a):
        """ (self, num) -> (VectorFunction)
        Defines scalar division of a VectorFunction.
        """
        if type(a) not in [float, int]:
            raise TypeError("A VectorFunction can only be divided by a scalar.")
        
        new_data = []
        def make_new(i):
            def temp(x):
                return 1/a*self[i](x)
            return temp

        for i in range(len(self)):
            new_data.append(make_new(i))
        return VectorFunction(new_data)
    

    def __add__(self, v2):
        """ (self, VectorFunction) -> (VectorFunction)
        Defines addition of VectorFunctions.
        """
        if not type(v2) is VectorFunction:
            raise TypeError("Can only add VectorFunctions together.")
        if len(self) != len(v2):
            raise ValueError("Can only add VectorFunctions of the same length.")

        new_data = []
        def make_new(i):
            def temp(x):
                return self[i](x) + v2[i](x)
            return temp

        for i in range(len(self)):
            new_data.append(make_new(i))
        return VectorFunction(new_data)


    def __sub__(self, v2):
        """ (self, VectorFunction) -> (VectorFunction)
        Defines subtraction of VectorFunctions.
        """
        if not type(v2) is VectorFunction:
            raise TypeError("Can only subtract VectorFunctions together.")
        if len(self) != len(v2):
            raise ValueError("Can only subtract VectorFunctions of the same length.")

        new_data = []
        def make_new(i):
            def temp(x):
                return self[i](x) - v2[i](x)
            return temp

        for i in range(len(self)):
            new_data.append(make_new(i))
        return VectorFunction(new_data)
    
    
    def magnitude(self):
        """ (self) -> (func)
        Returns a function that finds the magnitude of the self VectorFunction when called.
        """
        def new(x):
            tot = 0
            for func in self.data:
                tot += func(x)**2
            return math.sqrt(tot)
        return new
    
    
    def angle(self, x, vec):
        """ (self, num, Vector) -> (num)
        Finds the angle between the vec Vector and the self VectorFunction when evaluated at x.
        """
        return vec.angle(self(x))


    def scalar_proj(self, x, vec):
        """ (self, num, Vector) -> (num)
        Finds the scalar projection of the self VectorFunction when evaluated at x onto the vec Vector.
        """
        return self(x).scalar_proj(vec)
    

    def vector_proj(self, x, vec):
        """ (self, num, Vector) -> (num)
        Finds the vector projection of the self VectorFunction when evaluated at x onto the vec Vector.
        """
        return self(x).vector_proj(vec)
    
    
    def unit(self, x):
        """ (self, num) -> (Vector)
        Returns the unit Vector in the direction of the self VectorFunction evaluated at x.
        """
        return self(x).unit()


    def parallel(self, x, vec):
        """ (self, num, Vector) -> (Vector)
        Returns True if the self VectorFunction evaluated at x is parallel to the vec Vector. False otherwise.
        """
        return self(x).parallel(vec)


    def antiparallel(self, x, vec):
        """ (self, num, Vector) -> (Vector)
        Returns True if the self VectorFunction evaluated at x is anti-parallel to the vec Vector. False otherwise.
        """
        return self(x).antiparallel(vec)
    

    def opposite(self, x, vec):
        """ (self, num, Vector) -> (Vector)
        Returns True if the self VectorFunction evaluated at x is opposite to the vec Vector. False otherwise.
        """
        return self(x).opposite(vec)
    

    def perpendicular(self, x, vec):
        """ (self, num, Vector) -> (Vector)
        Returns True if the self VectorFunction evaluated at x is perpendicular to the vec Vector. False otherwise.
        """
        return self(x).perpendicular(vec)

    
    
    def arc_length(self, start, stop, steps=100):
        """ (self, num, num, int) -> (num)
        Approximates the arc length of the self VectorFunction from start to stop using the trapezoidal integral approximaiton.
        """
        return trapezoid_integral(self.magnitude, start, stop, steps)
    

    def eval_derivative(self, x, error=10**-5):
        """ (self, num, num) -> (Vector)
        Approximates the derivative of the self VectorFunction at x with an accuracy given by error.
        """
        new_data = []
        for func in self:
            new_data.append(derivative(func, x, error))
        return Vector(new_data)

    
    def tangent(self, x, error=10**-5):
        """ (self, num, num) -> (Vector)
        Approximates the unit tangent Vector of the self VectorFunction at x with an accuracy given by error.
        """
        deriv = self.eval_derivative(x)
        return deriv/deriv.magnitude()
    

    def normal(self, x, error=10**-5):
        """ (self, num, num) -> (Vector)
        Approximates the unit normal Vector of the self VectorFunction at x with an accuracy given by error.
        """
        tang = self.tangent(x, error)
        return tang/tang.magnitude()
    

    def binormal(self, x, error=10**-5):
        """ (self, num, num) -> (Vector)
        Approximates the binormal Vector of the self VectorFunction at x with an accuracy given by error. Only defined in 3D.
        """
        norm = self.normal(x, error)
        tang = self.tangent(x, error)
        return tang.cross_prod(norm)


# <h3>Statistics</h3>

# In[12]:


def mean(data):
    """ (list) -> (float)
    Returns the mean of the values in data.
    
    >>> ex = [0.1, 0.4, 0.6, 0.8, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 1.9, 2.0, 2.2, 2.6, 3.2]
    >>> mean(ex)
    1.5
    """
    return sum(data)/len(data)


def standard_dev(data, ave=None):
    """ (list, float/None) -> float
    Returns the standard deviation of the values in data.
    
    >>> ex = [0.1, 0.4, 0.6, 0.8, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 1.9, 2.0, 2.2, 2.6, 3.2]
    >>> standard_dev(ex)
    0.8434622525214579
    """
    if ave is None: #an average is not given and we must calculate it
        ave = mean(data) #finds the average of data
    #otherwise an average is already given as input
    
    #The following code computes the standard deviation of data
    std = 0
    for entry in data:
        std += (entry - ave)**2
        
    return math.sqrt(std/(len(data)-1))


def variance(data):
    """ (list) -> (float)
    Returns the variance of the values in data.
    """
    return standard_dev(data)**2


def standard_error(data, std=None):
    """ (list, float/None) -> (float)
    Returns the standard error of the values in data.
    
    >>> ex = [0.1, 0.4, 0.6, 0.8, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 1.9, 2.0, 2.2, 2.6, 3.2]
    >>> standard_error(ex)
    0.21778101714468007
    """
    if std is None:
        std = standard_dev(data)
    return std/math.sqrt(len(data))


def weighted_mean(data, errors):
    """ (list, list) -> (float, float)
    Returns the weighted mean of the entries of data, their weights are given by the inverse
    square of their uncertainties.
    Also returns the weighted mean's error.
    """
    weights = []
    for entry in errors: #the weight of a data point is the inverse square of its error
        weights.append(1/entry**2)
    
    tot = 0
    for i in range(len(data)):
        tot += data[i]*weights[i]
    
    #weighted mean
    final_mean = tot/sum(weights)
    
    #error in the weighted mean
    weighted_error = 1/math.sqrt(sum(weights))
    
    return final_mean, weighted_error


def percent_error(actual, expected):
    """ (float, float) -> (float)
    Returns the percent error of an experimentally determined value.
    """
    return abs((actual - expected)/expected)*100


# In[68]:


"""
The following code provides function suseful in determining linear fits to data as well as some ways of
testing the quality of the fit. 
"""


def linear_fit(x, y):
    """ (list list) -> (float, float, float, float)
    Returns the (slope, intercept, slope uncertainty, intercept uncertainty) of the linear fit of data y against data x.
    x and y have the same length.
    """
    N = len(x) #len(x)=len(y)

    if N != len(y):
        raise ValueError('x and y must have the same length.')

    #The following code finds the necessary sums of data needed to find a linear fit
    sum_x = sum(x)
    sum_y = sum(y)
    
    sum_x_squared = 0
    i = 0
    while i < N:
        sum_x_squared += x[i]**2
        i += 1
    
    sum_xy = 0
    i = 0
    while i < N:
        sum_xy += x[i]*y[i]
        i += 1
    
    #This is the denominator of many equations that are used in determining a linear fit.
    denominator = (sum_x_squared*N) - (sum_x)**2
    
    #slope
    m = ((N*sum_xy) - (sum_x*sum_y))/denominator
    
    #intercept
    c = ((sum_x_squared*sum_y) - (sum_x*sum_xy))/denominator
    
    #common uncertainty
    summation = 0
    i = 0
    while i < N:
        summation += (y[i] - m*x[i] - c)**2
        i += 1
    commonU = math.sqrt(summation/(N-2))
    
    #slope uncertainty
    mU = commonU*math.sqrt(N/denominator)
    
    #intercept uncertainty
    cU = commonU*math.sqrt(sum_x_squared/denominator)
    
    return m, c, mU, cU



def weighted_linear_fit(x, y, error):
    """ (list, list, list) -> (float, float, float, float)
    Determines the weighted least squares fit of a data set y against x with non-
    uniform error bars given by error.
    """
    weight = []
    for i in error:
        weight.append(1/i**2) #the weight of a given point is the inverse square of its error
    
    w = sum(weight) #sum of all the weights
    
    w_x = 0 #sum of all weights times their respective x point
    for i in range(len(weight)):
        w_x += weight[i]*x[i]
    
    w_y = 0 #sum of all weights times their respective y points
    for i in range(len(weight)):
        w_y += weight[i]*y[i]
    
    w_x_y = 0 #sum of all weights times their respective x and y points
    for i in range(len(weight)):
        w_x_y += weight[i]*x[i]*y[i]
    
    w_x_square = 0 #sum of all weights times their respective x points squared
    for i in range(len(weight)):
        w_x_square += weight[i]*(x[i]**2)
    
    delta = w*w_x_square-(w_x**2) #term found in the denominator of many equations used in finding the fit
    
    m = (w*w_x_y - w_x*w_y)/delta
    
    c = (w_x_square*w_y - w_x*w_x_y)/delta
    
    mU = math.sqrt(w_x_square/delta)
    
    cU = math.sqrt(w/delta)
    
    return m, c, mU, cU


def residuals(x, y, fit):
    """ (list, list, function, float) -> (list)
    Finds the residuals of a best fit single-variable function with uniform error and
    returns their y-coordiantes.
    """
    return [y[i]-fit(x[i]) for i in range(len(x))]


def normalised_residuals(x, y, fit, error):
    """ (list, list, function, list) -> (list)
    Finds the residuals of a best fit single-variable function with non-uniform error and
    returns their y-coordiantes. The error array is the standard error of the predicted values
    at each point in x.
    """
    return [(y[i]-fit(x[i]))/error[i] for i in range(len(x))]


def chi_square(x, y, fit, error):
    """ (list, list, list) -> (float)
    Returns the Chi-square value of a function, given by fit, fitted against x & y values with associated
    (not necessarily uniform) uncertainties given by error.
    """
    chi = 0
    for i in range(len(x)):
        chi += ((y[i] - fit(x[i]))/error[i])**2
    
    return chi


def chi_square_poisson(observed, expected):
    """ (list, list) -> (float)
    Returns the Chi-square value of a discrete function given by a Poisson distribution. Observed is a list of the
    observed number of counts for given intervals. expected is a list of the expected number of counts for given intervals.
    
    >>> chi_square_poisson([16, 18, 16, 14, 12, 12], [16, 16, 16, 16, 16, 8])
    3.5
    """
    chi = 0
    for i in range(len(observed)):
        chi += (observed[i] - expected[i])**2/expected[i]
    
    return chi


def durbin_watson(res):
    """ (list) -> (float)
    Returns the Durbin-Watson statistic which uses the residuals to test the fit of a function.
    D=0 : systematically correlated residuals
    D=2 : randomly distributed residuals that follow a Gaussian distribution
    D=4 : systematically anticorrelated residuals
    """
    numerator = 0
    for i in range(1, len(res)):
        numerator += (res[i] - res[i-1])**2
    
    denominator = 0
    for i in range(len(res)):
        denominator += res[i]**2
    
    return numerator/denominator


def rms(x, y, fit):
    """ (list, list, function) -> (float)
    Finds the root mean square of the fit to x and y data.
    """
    res_sqr = [r**2 for r in residuals(x, y, fit)]
    return math.sqrt(mean(res_sqr))
    


# <h3>Data Structures</h3>

# In[3]:


class Node:
    """
    A class to represent pointers in a linked list, queue, stack, deques, etc.

    Attributes
    ------
    data : any type
        The information contained in a Node.
    next : Node
        The following Node in a data structutre.
    previous : Node
        The previous Node in a data structutre.
    """

    def __init__(self, data=None):
        """ (self, any type) -> (Node)
        Overloading the constructor method for defining a new instance of the Node class.
        """
        self.data = data
        self.next = None
        self.previous = None


# In[4]:


class Stack:
    """
    A class to implement a Stack data structure. Information is added and removed from the top of the Stack in a last-in-first-out policy.

    Attributes
    -------
    head : Node
        The top of the Stack, the newest information added.

    Methods
    -------
    append(any type):
        Appends a new Node to the top of the Stack.
    peek():
        Returns the top value of the Stack.
    pop():
        Returns the top value of the Stack and removes that Node.
    """
    def __init__(self, data=None):
        """ (self, any type) -> (Node)
        Overloading the constructor method for defining a new instance of the Stack class.
        """
        if data is None:
            self.head = None
        else:
            self.head = Node(data)


    def __len__(self):
        """ (self) -> (int)
        Returns the number of elements in a Stack.
        """
        if not self.head: #if self is empty
            return 0
        
        i = 0 #count
        current = self.head #start counting at head
        while current is not None:
            i += 1
            current = current.previous
        return i

    
    def __str__(self):
        """ (self) -> (str)
        Return a meaningful string representation of the self Stack.
        """
        if self.head is None:
            return "||"
        
        string = str(self.head.data)
        current = self.head.previous
        while current is not None:
            string = str(current.data) + ", " + string
            current = current.previous
        return "|" + string + "|"



    def append(self, data):
        """ (self, any type) -> ()
        Adds a new Node with data to the top of the self Stack.
        """
        if self.head is None: #Stack is empty
            self.head = Node(data)
        else:
            self.head.next = Node(data)
            self.head.next.previous = self.head
            self.head = self.head.next
    

    def pop(self):
        """ (self) -> (any type)
        Returns the value at the top of the self Stack and removes that Node.
        """
        if self.head is None: #Stack is empty
            return None

        value = self.head.data #value at top of stack

        if len(self) >= 2: #self has more than 1 elements
            #deleting the top Node and setting second top Node to head
            self.head = self.head.previous
            del(self.head.next)
            self.head.next = None
        else: #self has exactly one element
            self.head = None

        return value

    
    def peek(self):
        """ (self) -> (any type)
        Returns the value at the top of the self Stack.
        """
        if self.head is None: #Stack is empty
            return None
        return self.head.data #value at top of stack


# In[15]:


class Queue:
    """
    A class to implement a Queue data structure. Information is added to the end of the Queue and removed from the front in a first-in-first-out policy.

    Attributes
    -------
    front : Node
        The front of the Queue, this is the entry we take out first.
    end : Node
        The end of the Queue, this is the most recently added entry.
    
    Methods
    -------
    peek():
        Returns the value at the front of the Queue.
    pop():
        Returns the value at the front of the Queue and removes this entry.
    prepend(any type):
        Adds a new entry to the end of the Queue.
    """
    
    def __init__(self, data=None):
        """ (self, any type) -> (Queue)
        Overloading the constructor method for defining a new instance of the Queue class.
        """
        if data is None: #empty Queue
            self.front = None
            self.end = None
        else:
            self.front = Node(data)
            self.end = self.front
        
    
    def __len__ (self):
        """ (self) -> (int)
        Returns the number of entries in the self Queue.
        """
        if self.front is None: #empty Queue
            return 0
        else:
            i = 0 #count
            current = self.front #start
            while current is not None:
                i += 1
                current = current.previous
            return i
    

    def __str__(self):
        """ (self) -> (str)
        Return a meaningful string representation of the self Queue.
        """
        if self.front is None: #empty Queue
            return ">>"

        string = str(self.end.data)
        current = self.end.next
        while current is not None:
            string = string + ", " + str(current.data) 
            current = current.next
        return ">" + string + ">"


    
    def pop(self):
        """ (self) -> (any type)
        Returns the data from the Node at the front of the Queue and removes that entry.
        """
        if self.front is None: #Queue is empty
            return None

        value = self.front.data #value at front of Queue

        if len(self) >= 2: #self has more than 1 element
            self.front = self.front.previous
            del(self.front.next)
            self.front.next = None
        else: #self has exactly one element
            self.front = None
            self.end = None

        return value



    def peek(self):
        """ (self) -> (any type)
        Returns the data from the Node at the front of the Queue.
        """
        if self.front is None: #Queue is empty
            return None
        return self.front.data


    def prepend(self, data):
        """ (self, any type) -> ()
        Adds a new Node with data to the end of the Queue.
        """
        if self.front is None: #empty Queue
            self.front = Node(data)
            self.end = self.front
        else:
            new = Node(data)
            new.next = self.end
            self.end.previous = new
            self.end = new


# In[24]:


class Deque(Queue):
    """
    A class to implement the Deque data structure. Information is added to the front and the ends of the Deque. Inherits from the Queue class.

    Attributes
    -------
    front : Node
        The front of the Deque, this is the entry we take out first.
    end : Node
        The end of the Deque, this is the most recently added entry.
    
    Methods
    -------
    append(any type):
        Adds a new entry to the front of the Deque.
    peek():
        Returns the value at the front of the Deque.
    pop():
        Returns the value at the front of the Deque and removes this entry.
    prepend(any type):
        Adds a new entry to the end of the Deque.
    prepeek():
        Returns the value at the end of the Deque.
    prepop():
        Returns the value at the end of the Deque and removes this entry.
    """

    def __str__(self):
        """ (self) -> (str)
        Return a meaningful string representation of the self Deque.
        """
        if self.front is None: #empty Deque
            return "><><"

        string = str(self.end.data)
        current = self.end.next
        while current is not None:
            string = string + ", " + str(current.data) 
            current = current.next
        return "><" + string + "><"

    def append(self, data):
        """ (self, any type) -> ()
        Adds a new Node with data to the front of the Deque.
        """
        if self.front is None: #empty Deque
            self.front = Node(data)
            self.end = self.front
        else:
            new = Node(data)
            new.previous = self.front
            self.front.next = new
            self.front = new


    def prepeek(self):
        """ (self) -> (any type)
        Returns the data from the Node at the front of the Deque.
        """
        if self.end is None: #Deque is empty
            return None
        return self.end.data
    

    def prepop(self):
        """ (self) -> (any type)
        Returns the data from the Node at the front of the Deque and removes that entry.
        """
        if self.end is None: #Deque is empty
            return None

        value = self.end.data #value at end of Deque

        if len(self) >= 2: #self has more than 1 element
            self.end = self.end.next
            del(self.end.previous)
            self.end.previous = None
        else: #self has exactly one element
            self.front = None
            self.end = None

        return value


# <h3>Searching & Sorting</h3>

# In[1]:


def binary_search(sorted_list, term):
    """ (list, num) -> (int)
    Searches a sorted list for a given element and returns the index corresponding to this element.
    
    >>> b =[1, 2, 4, 6, 10, 13, 14, 21, 30]
    >>> binary_search(b, 2)
    1
    >>> binary_search(b, 1)
    0
    >>> binary_search(b, 30)
    8
    >>> binary_search(b, 7)
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
    <ipython-input-3-98b7c6a02ae0> in <module>
    ----> 1 binary_search([1, 2, 4, 6, 10, 13, 14, 21, 30], 7)

    <ipython-input-1-ee09777e1120> in binary_search(sorted_list, term)
        21             lower = midpoint + 1
        22 
    ---> 23     raise ValueError("Element is not in given list.")

    ValueError: Element is not in given list.
    """
    #initial conditions: lower bound is first index, upper bound is last
    lower = 0
    upper = len(sorted_list)-1
    
    while lower <= upper: #if lower > upper we exhausted our search
        midpoint = (lower+upper)//2
        if sorted_list[midpoint] == term: #found term
            return midpoint
        elif sorted_list[midpoint] > term: #guess is too high, decrease upper bound
            upper = midpoint - 1
        elif sorted_list[midpoint] < term: #guess is too low, increase lower bound
            lower = midpoint + 1
        
    raise ValueError("Element is not in given list.")


# <h3>Misc.</h3>

# In[14]:


def factorial(n):
    """(int) -> (int)
    Returns the factorial of integer n."""
    if not type(n) is int:
        raise TypeError("Input must be a positive integer.")
    if n < 0:
        raise ValueError("Input must be a positive integer.")
    if n == 0: #special case
        return 1

    tot = 1 #answer
    current = 1 #start at 1
    while current <= n: #iterate from 1 to n
        tot *= current
        current += 1
    return tot


# In[14]:


def series(a_n, start, stop):
    """ (func, int, int) -> (float)
    Returns an approximation of the series defined by the sequence (a_n). 
    """
    tot = 0 #answer
    i = start #initial conditions
    while i <= stop: #iterate from start to stop
        tot += a_n(i) #value of a_n term at i
        i += 1 
    return tot


# In[56]:


if __name__ == '__main__':
    import doctest
    import time

    #doctest.testmod()

    #start = time.time()
    #m1 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    #print(m1.transpose())
    #print(round(time.time()-start, 10))

    #start = time.time()
    #func2()
    #print('func2 takes:', time.time()-start)

