#===================== Package import =======================
from prody import * 
import numpy as np
import warnings
import math
import time
import scipy.misc
import matplotlib.cm as cm
import h5py
from os import listdir,remove
from os.path import isfile, join
from multiprocessing import Pool

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Parameters:
            atomGroup            -> ProDy atomGroup structure
            eachChain (Optional) -> Selector for changing if return value should be a list of lists containing each 
                                    protein chain disconsidered residues indices(=TRUE) or one list containing all
                                    indices(=FALSE). Default value is False.
            
Return Values: 
            unwantedResidues     -> List  containing residues indices that shouldn't be considered on data
                                    transformation steps. Return value format depends on eachChain paramater 
                                    Value
Notes:
    It is important to notice that the indices returned by this function
    aren't usable as ProDy selection flags due to ProDy's change on indices
    values during parse function.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def findUnwantedResidues(atomGroup, eachChain = False):
    
    unwantedResidues = []

    for chainIdx in np.unique(atomGroup.select('stdaa').getChindices()):

        #List for handdling residues indexes that should be disconsidered
        chainUnwantedResidues = []

        #Select Chain Atoms and get only the standart amino acids (Data Cleaning)
        chainAtoms = atomGroup.select('chindex ' + str(chainIdx) + ' stdaa')
        
        #In some cases there might be only one chain on PDB file
        if(chainAtoms == None):
            chainAtoms = atomGroup.select('stdaa')
        
        #Check for problems on chain residues (occupancy < 1 or backbone missing atoms)
        
        #Occupancy check
        unwantedAtoms = chainAtoms.select('occupancy < 1')

        if(unwantedAtoms != None):
            for atom in unwantedAtoms:
                chainUnwantedResidues.append(atom.getResindex())
                
        #Return value assemble
        if(eachChain):
            unwantedResidues.append(np.unique(chainUnwantedResidues))
        else:
            aux = np.unique(chainUnwantedResidues).tolist()
            unwantedResidues = unwantedResidues + aux
        
		#DEBUG
        #warnings.warn("Remember that the indices returned by this function should not be used as select flags due to the alteration of their values by ProDy on parsing")

    return unwantedResidues

'''
Parameters:
            atomGroup               -> ProDy atomGroup structure
            filterList  (Optional)  -> List  containing residues indices to be disconsidered on each chain.
            bothResults (Optional)  -> Selector for returning not only the filtered result value but the
                                       result value when filterList = []
               
Return Values: 
            chainList               -> List of list containing the atoms of each chain from passed atomGroup disconsidering 
                                       all data present on filterList.Each list consists on an entire protein chain.
            notFilteredChainList    -> Same as chainList data, but without filtering any data.
'''
def groupAtomsByChain(atomGroup, filterList = [], bothResults = False):
    
    chainList = []
    
    if(bothResults):
        notFilteredChainList = []
    
    #Create list of chain atoms list
    for chainIdx in np.unique(atomGroup.getChindices()):
        
        #Create an temporary auxiliar atom list
        auxAtomList = []
        if(bothResults):
            auxNotFilteredChainList = []
              
        #Insert all atoms on chain list
        for atom in atomGroup.select('chindex ' + str(chainIdx)):
            #Filter unwanted atoms
            if(atom.getResindex() not in filterList):
                auxAtomList.append(atom)
            if(bothResults):
                auxNotFilteredChainList.append(atom)
                
        chainList.append(auxAtomList)
        
        if(bothResults):
            notFilteredChainList.append(auxNotFilteredChainList)
        
    if(bothResults):
        return chainList,notFilteredChainList
    else:
        return chainList
    
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Parameters:
            atomGroup            -> List of ProDy's Atom class
            
Return Values: 
            np.unique(resIdx)    -> List containing all residue indices which all atoms in atomGroup belongs     

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def atomGroupToResIdx(atomGroup):
    resIdx = []
    for atom in atomGroup:
        resIdx.append(atom.getResindex())
    return np.unique(resIdx)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Parameters:
            dim                  -> Relevant space dimension
            center               -> Relevant space center coordinates
            atom                 -> Atom coordinates to be checked if is in bounds
            sphere               -> Check for spheric space unless standart cubic space
            
Return Values: 
            True/False           -> True if in bounds or False if out of it

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def inRelevant3DSpace(dim,center,atom,spheric=False):
    
    #It's is needed to calculate the cube diagonal and divide by 2 because the reference that is being used
    #is the cube center. It's being used this value to check if the atom is in bounds due to the future matrix
    #rotation process that if it's not considered a sphere area some atoms which should be considered as relevant
    #may be left behind.
    
    if(spheric):
        #Get cube diagonal value for considering spheric space 
        bound = (dim * math.sqrt(3))/2
    else:
        bound = dim
    
    #Check if atom is out of bounds on x axis
    if((atom[0] >= (center[0] + bound)) or (atom[0] <= (center[0] - bound))):
        return False #It's out!
    #Check if atom is out of bounds on y axis
    elif((atom[1] >= (center[1] + bound)) or (atom[1] <= (center[1] - bound))):
        return False #It's out!
    #Check if atom is out of bound on z axis
    elif((atom[2] >= (center[2] + bound)) or (atom[2] <= (center[2] - bound))):
        return False #It's out!
    else:
        return True #It's in!

    
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Parameters:
            atomGroup                 -> Atom group which selection will be done
            centerResIdx              -> Centralized residue index
            
Return Values: 
           atomElements               -> Numpy array containing each atom element
           atomCoordinates            -> Numpy array containing each atom x,y,z coordinates
           resName                    -> Residue Name which selected atoms belongs
           alphaCarbonCoord           -> Centralized residue backbone alpha carbon coordiantes
           carbonCoord                -> Centralized residue backbone carbon coordinates
           oxigenCoord                -> Centralized residue backbone oxigen coordinates
           nitrogenCoord              -> Centralized residue backbone nitrogen coordiantes

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#Check if is an atom from centralized residue
def getElementsCoordsResname(atomGroup,centerResIdx):

    atomElements = []
    atomCoordinates = []
    alphaCarbonCoord = carbonCoord = oxigenCoord = nitrogenCoord = None
    resName = None
    
    #Mount list of coordinate tuples for future transformations (Translation and Rotation)
    
    for atom in atomGroup:
        if(atom.getResindex() == centerResIdx):

            atomName = atom.getName()  

            #Check if is a backbone atom 
            if(atomName in ['N','CA','C','O']):
                #Store centralized residue backbone atoms coordinates -> They are used to find the multiplication matrix
                if(atomName == 'CA'):
                    alphaCarbonCoord = atom.getCoords()
                    resName = atom.getResname()
                elif(atomName == 'C'):
                    carbonCoord = atom.getCoords()
                elif(atomName == 'O'):
                    oxigenCoord = atom.getCoords()
                else:
                    nitrogenCoord = atom.getCoords()
                    
                atomElements.append(atom.getElement())

                #Insert atom coordinates tuple on coordinates list
                atomCoordinates.append(atom.getCoords())

        #Atom Not from centralized residue
        else:
            atomElements.append(atom.getElement())

            #Insert atom coordinates tuple on coordinates list
            atomCoordinates.append(atom.getCoords())
        
    #Transform data lists on Numpy array for optimized operations
    atomCoordinates = np.asarray(atomCoordinates,dtype=np.float64)
    atomElements = np.asarray(atomElements,dtype=np.str)
  
    return atomElements,atomCoordinates,resName,alphaCarbonCoord,carbonCoord,oxigenCoord,nitrogenCoord
	
	
"""
Calculate the normalized coordinates values for C,CA,O atoms.

Parameters
----------
C : tuple
    (x,y,z) tuple containing x,y,z coordinates from backbone Carbon(C) atom.
CA : tuple
    (x,y,z) tuple containing x,y,z coordinates from backbone Alpha Carbon(CA) atom.
O : tuple
    (x,y,z) tuple containing x,y,z coordinates from backbone Oxygen (O) atom.

Returns
-------
M : np.mat(...,  dtype = np.float64)
    (N,D) matrix where N are each atom and D each axis.
"""

def calculateNormalizedMatrix(C,CA,O):
    
    #Calculate normalized coordinates values
    
    '''
       Normalized atoms expected coordinates

            axis Y
                    │       
                    │      
                 CA (0,CAy,0)    
                    |    
                    |  
                    |  
                    |                  axis X
            (0,0,0) C—————————————————— 
                    │\ 
                    │ \
                    │  \
                    │   \
                    │    O (Ox,Oy,0)
    '''   
    
    CAx,CAy,CAz = CA[0],CA[1],CA[2]
    Cx,Cy,Cz = C[0],C[1],C[2]
    Ox,Oy,Oz = O[0],O[1],O[2]
    
    #---- Discover OCX angle ----#
    
    #DCA = Distance from C to CA
    DCA = math.sqrt(CAx**2 + CAy**2 + CAz**2)

    #DO = Distance from C to O
    DO = math.sqrt(Ox**2 + Oy**2 + Oz**2)

    #DCAO = Distance from CA to O
    DCO = math.sqrt((CAx-Ox)**2 + (CAy-Oy)**2 + (CAz-Oz)**2)

    #Calculate Alpha Angle
    sigma = math.acos((DO**2 + DCA**2 - DCO**2)/(2* DO * DCA))
    alpha = math.pi/2 - sigma

    #---- Calculate Cy' , Ox' and Oy' values ----#

    CAyf = DCA
    Oxf = DO * math.cos(alpha)
    Oyf = DO * math.sin(alpha)
    
    M = np.mat([[0,0,0],[0,CAyf,0],[Oxf,Oyf,0]], dtype = np.float64)
    
    return M


"""
Find rotation matrix necessary to align one matrix in another using Kabsch Algorithm
https://en.wikipedia.org/wiki/Kabsch_algorithm

Parameters
----------
A : np.mat(... ,dtype = np.float64)
    (N,D) matrix that will be aligned, where N are each atom and D each axis.
B : np.mat(... ,dtype = np.float64)
    (N,D) alignment reference matrix, where N are each atom and D each axis.
translate : boolean
    Use centroids to translate matrix position during alignment. Necessary if the translation
    hasn't been done yet.

Returns
-------
R : np.mat(... ,dtype = np.float64)
    Matrix containing all rotation values necessary for the alignment.
t : np.mat(... ,dtype = np.float64)
    Matrix containing all translation values necessary for the alignment.
"""


def findRotationMatrix(A, B, translate=False):
    assert len(A) == len(B)
    
    #Find rotation matrix using Kabsch Algorithm
    
    
    N = A.shape[0]; # total points
    
    if translate:
    
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))

        # dot is matrix multiplication for array
        H = np.transpose(AA) * BB
        
    else :
        
        H = np.transpose(A)* B
        
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("Reflection detected") #DEBUG
        Vt[2,:] *= -1
        R = Vt.T * U.T
    
    if translate:
        
        t = -R*centroid_A.T + centroid_B.T
        return R, t
    
    else :
        
        return R
    

"""
Rotate matrix A using R as rotation matrix

Parameters
----------
A : np.mat(... ,dtype = np.float64)
    (N,D) matrix that will be rotated, where N are each atom and D each axis.
R : np.mat(... ,dtype = np.float64)
    Rotation matrix that will be used.

Returns
-------
A_R : np.mat(... ,dtype = np.float64)
    Matrix A rotated with rotation matrix R values.
  
"""

def rotate(A,R):

    A_R = (R*A.T)
    A_R = A_R.T
    
    return A_R

"""
Calculate root-mean squared error (rmse)

Parameters
----------
A : np.mat(... ,dtype = np.float64)
    (N,D) Expected matrix, where N are each atom and D each axis.
B : np.mat(... ,dtype = np.float64)
    (N,D) Obtained matrix, where N are each atom and D each axis.
numOfDots : integer
    D size of A and B matrices
    
Returns
-------
rmse : np.float64
    root-mean squared error. If value near 0, matrices are aligned.
  
"""

def calcRmse(A,B,numOfDots):
     
    # Find the error
    err = B - A
    err = np.multiply(err, err)
    err = np.sum(err)
    rmse = math.sqrt(err/numOfDots)
    
    return rmse
	
"""
Transform atom element to it's respective discretization grid index.

Parameters
----------
element : char
    Should be one of the following list: 
    Carbon(C), Nitrogen(N), Oxygen(O), Sulfur(S).
     
Returns
-------
Integer value
    Value respective to the grid where this elements data will be stored.
    0 = Carbon, 1 = Oxygen, 2 = Nitrogen, 3 = Sulfur, -1 = Invalid element
"""

def elementToIndex(element):
    if element == 'C':
        return 0
    elif element =='O':
        return 1
    elif element == 'N':
        return 2
    elif element == 'S':
        return 3
    else:
        #Invalid element
        return -1
    
    
"""
Check if 3D dot is inside volume.

Parameters
----------
xIdx : int
    Dot x axis value.
yIdx : int
    Dot y axis value.
zIdx : int
    Dot z axis value. 
gridDimension: int
    Discretization grid dimension.
Returns
-------
Boolean value
    True = In bound
    False = Out of bounds
    
"""

def inGridBounds(xIdx,yIdx,zIdx,gridDimension):
    if xIdx >= 0 and xIdx < gridDimension:
        if yIdx >= 0 and yIdx < gridDimension:
            if zIdx >= 0 and zIdx < gridDimension:
                #In bounds
                return True
    
    #Out of bounds
    return False


"""
Calculate atom relative position on discretization grid.

Parameters
----------
x : int
    Atom x axis value.
y : int
    Atom y axis value.
z : int
    Atom z axis value. 
element: char
    Atom element, should be one of the following list: 
    Carbon(C), Nitrogen(N), Oxygen(O), Sulfur(S).
gridScale: float
    Scale factor used on grid creation
gridDimension: int
    Discretization grid dimension.
    
Returns
-------
if index is valid
    tuple: (elementIdx,xIdx,yIdx,zIdx)
        elementIdx : int
            Element grid index
        xIdx : int
            Atom x axis grid relative index
        yIdx : int
            Atom y axis grid relative index
        zIdx : int
            Atom z axis grid relative index
else
    Integer value: -1


"""

def calcGridPosition(x,y,z,element,gridScale,gridDimension):

    #Calculate Grid "central" value for index value correction
    N = math.floor(gridDimension/2)
    xIdx = math.floor(x/gridScale) + N
    yIdx = math.floor(y/gridScale) + N
    zIdx = math.floor(z/gridScale) + N
    
    if inGridBounds(xIdx,yIdx,zIdx,gridDimension):

        #Identify element index
        elementIdx = elementToIndex(element)
    
        if elementIdx == -1:
            #raise Exception('Invalid Element ->' + element) DEBUG
            
            #Invalid Element, disconsider this atom
            return -1
            
        return (elementIdx,xIdx,yIdx,zIdx)
    
    else:
        
        #Index out of bound, disconsider this atom
        return -1
    
    
def startStorageFile(protName,resName,resIdx,storage_dir=''):
    
    filename = storage_dir + '_'.join([protName, str(resIdx).zfill(4), resName]) + '.hdf5'
    
    f = h5py.File(filename,'a')
    
    return f

def storeGrid(f,discGrid):
    
    f['grid'] = discGrid
    
def finishStorageFile(f):

    f.close()
	
def parallelTransf(name):
       
    #================ Set Configuration Variables ===============

    RS_DIM = 21
    #Value in ångströms

    #It's important to GRID_SCALE be a value which RS_DIM/GRID_SCALE = integer, otherwise the data discretizantion will not work!!

    GRID_SCALE = 0.7 #Value in ångströms

    #Discretization Gird dimesion definition
    GRID_DIM = RS_DIM/GRID_SCALE
    GRID_DIM = int(GRID_DIM)
    #Number of elements
    N_ELEMENTS = 4 # C,N,O,S

    voxelCoexistence = {}
    errors = []

    #================== Fetch file names =========================

    #Storage directory
    STORAGE_PATH = 'D:/top8000_discretized/'
    DATA_PATH = 'top8000_hom50'
    
    #=================== Parse Data ==============================
    
    #Remove file extension

    print('Transformation Started ==>  ' + name)
    atomGroup = prody.parsePDB(DATA_PATH + '/' + name)
    name = name.split('.')[0]
    protName = name

    #=================== Data Cleaning ===========================
    try:
        #Get residues indices that should be disconsidered on further steps
        unwantedResindices = findUnwantedResidues(atomGroup,eachChain=False)
        
        #Group atoms by their chain index and filter unwanted residues indices (Effectively the data cleaning)
        toBeTransfAtomGroup,completeAtomGroup = groupAtomsByChain(atomGroup.select('stdaa'),filterList = unwantedResindices,bothResults=True)
        
        #BENCHMARK
        startTime = time.time()
        
        #Chains are being considered as standalone structures
     
        #================== Data Transformation==============================
        
        #For each chain
        for (toBeTransfchain,completeChain) in zip(toBeTransfAtomGroup,completeAtomGroup):
        
        	#Store chain residues indices containing only the ones whose shall be transformed
        	chainResIdx = atomGroupToResIdx(toBeTransfchain)
        
        	#For each residue in chain
        	for centerResIdx in chainResIdx:
			
        		#Data Transformation variables definitions
        		atomElements = []
        		atomCoordinates = []
        		alphaCarbonCoord = carbonCoord = oxigenCoord = nitrogenCoord = None
        		resName = None
        
        		#Get atom coordinates/elements data + remove centralized residue lateral chain atoms (We don't want to give our AI any clues of the residue :D)
        		atomElements,atomCoordinates,resName,alphaCarbonCoord,carbonCoord,oxigenCoord,nitrogenCoord = getElementsCoordsResname(completeChain,centerResIdx)
        		#DEBUG
        		#print('--------RES ',centerResIdx,'==>',resName,'----------')
        		
        		#Open storage file
        		f = startStorageFile(protName,resName,centerResIdx,STORAGE_PATH)
        		#------------------------------------Matrix Spacial Translation---------------------------------------------
        
        		#BENCHMARK
        		transTime = time.time()
        		# Translate coordinate list and centralized residue backbone atoms
        
        		#DEBUG
        		#print(alphaCarbonCoord,atomCoordinates)
        		#DEBUG
        		#print(centerCoord[0],centerCoord[1],centerCoord[2])
        
        		#Translate all atoms
        		atomCoordinates = np.subtract(atomCoordinates,carbonCoord)
        
        		#Translate centralized residue backbone atoms, positioning the carbon atom at 0,0,0 coordinates
        		carbonCoord = np.subtract(carbonCoord,carbonCoord)
        		oxigenCoord = np.subtract(oxigenCoord,carbonCoord)
        		#nitrogenCoord = np.subtract(nitrogenCoord,carbonCoord)
        		alphaCarbonCoord = np.subtract(alphaCarbonCoord,carbonCoord)
        
        		#ATOMS ALREADY INSERTED ON LIST
        
        		#BENCHMARK
        		endTransTime = time.time()
        		#DEBUG
        		#print('Translation time: ',endTransTime - transTime)
        
        		#-----------------------------------------------------------------------------------------------------------
        
        		#DEBUG
        		#print(atomCoordinates)
        
        		#------------------------------------Matrix Spacial Rotation------------------------------------------------
        
        		#BENCHMARK
        		rotIdentifTime = time.time()
        
        		atomCoordinates = np.mat(atomCoordinates,dtype=np.float64)
        
        		#Mount initial situation matrix
        		im = np.mat([carbonCoord,alphaCarbonCoord,oxigenCoord], dtype = np.float64)
        
        		#Calculate final situation matrix (normalized matrix)
        		fm = calculateNormalizedMatrix(carbonCoord,alphaCarbonCoord,oxigenCoord)
        
        		#Find rotation matrix. (As translation has already been done, translate = False)
        		rm = findRotationMatrix(im,fm,translate = False)
        
        		#Rotate
        		im_r = rotate(im,rm)
        
        		#Check rotation error
        		rmse = calcRmse(fm,im_r,3)
        		#DEBUG
        		#print('RMSE value should be near 0\n','RMSE:',rmse)
        
        		#BENCHMARK
        		endRotIdentifTime = time.time()
        		#DEBBUG
        		#print('Rotation Matrix Identification time: ',endRotIdentifTime - rotIdentifTime)
        
        		#BENCHMARK
        		rotTime = time.time()
        
        		#Rotate Structure based on centralized residue normalization
        		atomCoordinates = rotate(atomCoordinates,rm)
        
        		#BENCHMARK
        		endRotTime = time.time()
        		#DEBUG
        		#print('Structure Rotation time: ',endRotTime - rotTime)
        
        		#-----------------------------------------------------------------------------------------------------------
        
        		#---------------------------------------Data Discretization-------------------------------------------------
        
        		#BENCHMARK
        		discTime = time.time()
        
        		#Discretization Grid(discGrid)
        		#Stores 4 3D matrices, each one respective to one element
        		#[0]-> Carbon 
        		#[1]-> Oxygen 
        		#[2]-> Nitrogen 
        		#[3]-> Sulfur
        
        		discGrid = np.zeros(shape=(N_ELEMENTS,GRID_DIM,GRID_DIM,GRID_DIM),dtype=np.bool)
        
        		for coord,element in zip(atomCoordinates,atomElements):
        
        			r = calcGridPosition(coord[0,0],coord[0,1],coord[0,2],element,GRID_SCALE,GRID_DIM)
        
        			#Check if out is in bounds
        			if r != -1:
        
        				if discGrid[r[0],r[1],r[2],r[3]] == False:
        					discGrid[r[0],r[1],r[2],r[3]] = True
        				else:
        					#Check if atom voxel coexistance has already happened for this residue
        					if (protName +'_' + str(centerResIdx)) in voxelCoexistence:
        						voxelCoexistence[(protName +'_'+ str(centerResIdx))] += 1
        					else:
        						voxelCoexistence[(protName +'_'+ str(centerResIdx))] = 1
        
        		#BENCHMARK
        		endDiscTime = time.time()
        		#DEBUG
        		#print('Discretization Time: ',endDiscTime - discTime)
        
        		#Store Grid
        		storeGrid(f,discGrid)
        
				#Close protein storage file
        		finishStorageFile(f)
        
        endTime = time.time()
    except:
        #Close protein storage file
        if(f in locals()):
            finishStorageFile(f)
            errors.append(name)
            #Delete file with bad data
            remove(STORAGE_PATH + name + '.hdf5')

	#DEBUG
	#print('Total time: ',endTime - startTime)
    
	#DEBUG
    #print('Voxel Coexistences: ',voxelCoexistence)
	
    print('Transformation Finished ==>  ' + name)
    return voxelCoexistence , errors 
	
def teste(name):
	f = startProtFile(name,'D:/top8000_discretized/')
	finishProtFile(f)
	
if __name__== "__main__":
	#================ Set Configuration Variables ===============
	'''
	RS_DIM = 21
	#Value in ångströms

	#It's important to GRID_SCALE be a value which RS_DIM/GRID_SCALE = integer, otherwise the data discretizantion will not work!!

	GRID_SCALE = 0.7 #Value in ångströms

	#Discretization Gird dimesion definition
	GRID_DIM = RS_DIM/GRID_SCALE
	GRID_DIM = int(GRID_DIM)
	#Number of elements
	N_ELEMENTS = 4 # C,N,O,S

	voxelCoexistence = {}

	#================== Fetch file names =========================

	#Storage directory
	STORAGE_PATH = 'D:/top8000_discretized/'
	'''
	
	#path = 'Top8000_hom50'
	DATA_PATH = 'top8000_hom50'
	#Get file names
	fileNames = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]

	print(fileNames) #DEBUG
	print(len(fileNames))

	p = Pool(3)

	#voxelCoexistence = p.map(teste,fileNames)
	voxelCoexistence,errors = p.map(parallelTransf,fileNames)
	print('All Voxel Coexistences: ',voxelCoexistence)
	unique_errors = np.array(errors) 
	print(np.unique(unique_errors))
