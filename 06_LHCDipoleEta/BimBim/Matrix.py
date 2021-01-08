import scipy.sparse as spm
import numpy as np

#Define direct product of two matrices
def directProd(a,b):
  return spm.kron(a,b)
  
#Define dot product of abitrary number of matices)
def dotProd(a):
  out = a[0]
  for i in range(1,len(a)):
    out = out.dot(a[i])
  return out
  
 
#Get maximum <0.0 of a 2D array
def maxi_neg(a):
  maxi = 0.0
  index=0
  for i in range(len(a)):
    for j in range(len(a[0])):
      if maxi >= 0: 
        maxi = a[i][j]
        index = j
      if a[i][j]>maxi and a[i][j]<0.0: 
        maxi = a[i][j]
        index = j
  return maxi,index
 
#Remove negligible values from a sparse matrix
def remove_zeros(a):
  for i in range(len(a.data)):
      if abs(a.data[i])<1.0e-12: a.data[i]=0.0
  a.eliminate_zeros();

def diag(i,j):
  if i==j:
    return 1.0
  else:
    return 0.0

def idMatrix(u):
  return spm.eye(u,u)
  
#Compute eigenvalues of matrix a
#def eigval(a):  
#  return np.linalg.eigvals(a)
  
def transpose(a):
  return np.array(a).transpose()
  
def printMatrix(filename,a):
  if spm.issparse(a):
    dense = a.todense();
  else:
    dense = a;
  for i in range(dense.shape[0]):
    line = ""
    for j in range(dense.shape[1]):
      if np.iscomplexobj(dense[i,j]):
        line = line+"\t"+str(dense[i,j])
      else:
        line = line+"\t"+str("%E"%(dense[i,j]))
    line = line + "\n"
    filename.write(line)
    
def printVector(filename,a):
  line = ""
  shape = np.shape(a)
  size = shape[0]
  if len(shape) > 1 and shape[1] > shape[0]:
    size = shape[1]
  for i in range(size):
    line = line+"\t"+str(a[i])
  line = line + "\n"
  filename.write(line)
  
#Kepp only positive values
def keeppos(a,eps):
  if a<eps: 
    return 0.0
  else:
    return a
    
#Define matrix of distance between slices
def sdiff(nslice,nrings,posvec):
  distmat = [[keeppos(posvec[j]-posvec[i],1.0e-12) for j in range(nslice*nrings)] for i in range(nslice*nrings)] 
  return distmat
  
#set values to zer os if too small
def setzero(a,eps):
  if abs(a)<eps: 
    return 0.0
  else:
    return a
    
def sparse_insert(a,b,i_start=0,j_start=None):
    if j_start == None:
        j_start = i_start;
    if spm.isspmatrix_coo(b):
        for i,j,v in zip(b.row,b.col,b.data):
            a[i+i_start,j+j_start] = v;
    elif spm.isspmatrix_dok(b):
        for key in b.keys():
            a[key[0]+i_start,key[1]+j_start] = b.get(key);
    else:
        tmpb = smp.coo_matrix(b);
        for i,j,v in zip(tmpb.row,tmpb.col,tmpb.data):
            a[i+i_start,j+j_start] = v;
            
def sparse_add(a,b,i_start=0,j_start=None):
    if j_start == None:
        j_start = i_start;
    if spm.isspmatrix_coo(b):
        for i,j,v in zip(b.row,b.col,b.data):
            a[i+i_start,j+j_start] += v;
    elif spm.isspmatrix_dok(b):
        for key in b.keys():
            a[key[0]+i_start,key[1]+j_start] += b.get(key);
    else:
        tmpb = smp.coo_matrix(b);
        for i,j,v in zip(tmpb.row,tmpb.col,tmpb.data):
            a[i+i_start,j+j_start] += v;
