#author: Stefan Thalhammer
#date: 23.5.2018

import string, sys
from numpy import *
from numpy.linalg import *

class Image:
    
    def __init__(self, pc, normals):
        
        self.rows = rows = int(normals.shape[0])
        self.cols = cols = int(normals.shape[1])
        self.image = []

        self.normals = normals
        self.color = ones(normals.shape, dtype=uint8) * -1
        self.coords = reshape(pc, (rows, cols, 3))

        self.eclass_label = ones((rows, cols), dtype=uint8) * -1
        
        self.type = chararray((rows, cols))
        self.type[:] = 'unknown'
        self.is_boundary = full((rows, cols), False)

    def get_point(self,row,col):

        return self.coords[row][col]     
    
    def get_kxk_neighborhood(self,row,col,k):

        wid = int((k - 1) * 0.5)
        
        return self.coords[(row-wid):(row+wid+1), (col-wid):(col+wid+1), :]            

               
class UnionFind:
        
    def __init__(self):

        self.leader = {}        #dictionary that given key=label, returns leader of label's eclass
        self.group = {}         #given a leader, return the partition set it leads
        
    def add(self,a,b):
        '''add a and b to the object. if they're already in the same eclass do nothing. If both belong to different eclasses, merge 
        the smaller eclass into the larger eclass and delete the smaller, and make sure every former member of the smaller knows its
        new leader. If only one of a and b is in an eclass, but the singleton into the other's eclass. If neither have an eclass, just
        make a new one with them as members and a as the leader.'''
        
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return       #these aren't distinct groups, quit
                groupa = self.group[leadera]        
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):       #make a the larger set (just flip around)
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb        #a's group now includes everything in b; '|' is set-union in python
                del self.group[leaderb]
                for k in groupb:        #reassign leader for every member of b since it's been merged into a
                    self.leader[k] = leadera
            else:
                #if a's group isn't empty but b's group is, just stick b into a's group
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                #if a's group is empty but b's group isn't, stick a into b's group
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])
        
    def make_new(self,a):
        '''add a new singleton partition with a as the leader and sole member'''
        self.leader[a]=a
        self.group[a] = set([a])        #a now leads a new singleton set
                
def are_locally_coplanar(P, p_thresh):
    
    k = len(P[0])
    centroid = (sum(P, axis=(0, 1)))/(k*k)

    A = array([[0,0,0],[0,0,0],[0,0,0]])
    for i in range(0,k):
        for j in range(0,k):
            [p1,p2,p3] = P[i][j] - centroid
            A = A + array([[p1],[p2],[p3]])*array([p1,p2,p3])
                
    #local normal for this neighborhood is the eigenvector for the smallest eigenvalue of the matrix A, 
    #where A is the sum of the difference vectors between each point and the centroid.
    
    eigs = eigh(A)
                                 #We use eigh to only get real eigenvalues.
    eigenvalues = eigs[0]       #array of eigenvalues of A, unsorted
    eigenvectors = eigs[1]      #array of eigenvectors of A, normalized, ordered according to their corresponding eigenvalue in eigs[0]
    min_eval_index = argmin(eigenvalues)     #min_eval_index is the *index* of the smallest eigenvalue
            #this is going to be imaginary sometimes? but not if we use eigh(A) instead of eig(A)
    min_eigenval = eigenvalues[min_eval_index]  #smallest eigenvalue
    normal = eigenvectors[min_eval_index]     #The eigenvector associated with the smallest eigenvalue is the normal. (We don't actually need this...)     

    if(min_eigenval <= p_thresh): return True   #if the eigenvalue is small enough, these points are coplanar so return True.
    else: return False
    


