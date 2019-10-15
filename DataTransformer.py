#TO DO 
#0) GENERIC PATH FUNCTIONS SO THE MODULE CAN BE USED IN LINUX AND WINDOWS
#1) RENAME PRIVATE FUNCTIONS WITH _ BEFORE NAME
#2) FILE MANIPULATION SHOULB BE DONE BY ANOTHER MODULE no by DataTransforme
#3) PARALELLIZE ON CPU THE PROCESS
#4) TRY CATCH ON TRANSFORM FUNCTIONS IS ON WRONG PLACE, BECAUSE IT CAUSE THE LOOP TO TOTALLY STOP
#WHEN HAPPENING A EXCEPTION BUT IT SOULD ONLY STOP THAT ITERATION AND DELETE THE FILE RESPECTIVE TO
#THAT ITERATION
#4) TRANFORM FUNCTION RECEIVE FILEPATH AND FILENAME AS DIFFERENT PARAMETERS??

from prody import * 
import numpy as np
import warnings
import math
import time
import scipy.misc
import matplotlib.cm as cm
import h5py
import glob
from os import listdir,remove
from os.path import isfile, join

class DataTransformer():

    def __init__(self,rs_dim,grid_scale,n_elements):

        #================ Set Configuration Variables ===============

        #Relevant space dimensions definition 
        self.rs_dim = rs_dim #Value in ångströms

        #Grid scale factor definition
        self.grid_scale = grid_scale #Value in ångströms

        #Occupancy grid dimensions definition
        self.grid_dim = rs_dim/grid_scale
        self.grid_dim = int(self.grid_dim)
        self.n_elements = n_elements


    def findUnwantedResidues(self,atomGroup, eachChain = False):
    
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
            
        return unwantedResidues  

    def groupAtomsByChain(self,atomGroup, filterList = [], bothResults = False):
    
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

    def atomGroupToResIdx(self,atomGroup):
        resIdx = []
        for atom in atomGroup:
            resIdx.append(atom.getResindex())
        return np.unique(resIdx)

    def getElementsCoordsResname(self,atomGroup,centerResIdx):

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

    def calculateNormalizedMatrix(self,C,CA,O):
    
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

    def findRotationMatrix(self,A, B, translate=False):
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

    def rotate(self,A,R):

        A_R = (R*A.T)
        A_R = A_R.T
        
        return A_R

    def calcRmse(self,A,B,numOfDots):
     
        # Find the error
        err = B - A
        err = np.multiply(err, err)
        err = np.sum(err)
        rmse = math.sqrt(err/numOfDots)
        
        return rmse

    def elementToIndex(self,element):
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

    def inGridBounds(self,xIdx,yIdx,zIdx,gridDimension):
        if xIdx >= 0 and xIdx < gridDimension:
            if yIdx >= 0 and yIdx < gridDimension:
                if zIdx >= 0 and zIdx < gridDimension:
                    #In bounds
                    return True
        
        #Out of bounds
        return False

    def calcGridPosition(self,x,y,z,element,gridScale,gridDimension):

        #Calculate Grid "central" value for index value correction
        N = math.floor(gridDimension/2)
        xIdx = math.floor(x/gridScale) + N
        yIdx = math.floor(y/gridScale) + N
        zIdx = math.floor(z/gridScale) + N
        
        if self.inGridBounds(xIdx,yIdx,zIdx,gridDimension):

            #Identify element index
            elementIdx = self.elementToIndex(element)
        
            if elementIdx == -1:
                #raise Exception('Invalid Element ->' + element) DEBUG
                
                #Invalid Element, disconsider this atom
                return -1
                
            return (elementIdx,xIdx,yIdx,zIdx)
        
        else:
            
            #Index out of bound, disconsider this atom
            return -1

    def startStorageFile(self,protName,resName,resIdx,storage_dir=''):
    
        filename = storage_dir + '_'.join([protName, str(resIdx).zfill(4), resName]) + '.hdf5'
        
        f = h5py.File(filename,'a')
        
        return f

    def storeGrid(self,f,discGrid):
        
        f['grid'] = discGrid
        
    def finishStorageFile(self,f):

        f.close()

    def cleanDirectory(self,storage_dir):
        filenames = (glob.glob(storage_dir + '*'))
        
        for filename in filenames:
            remove(filename)

    def transform(self,filename,filepath,storagePath):
       
        voxelCoexistence = {}
        errors = []
        
        #=================== Parse Data ==============================
        
        #Remove file extension

        print('Transformation Started ==>  ' + filename)
        atomGroup = prody.parsePDB(filepath + '/' + filename)
        filename = filename.split('.')[0]
        protName = filename

        #=================== Data Cleaning ===========================
        try:
            #Get residues indices that should be disconsidered on further steps
            unwantedResindices = self.findUnwantedResidues(atomGroup,eachChain=False)
            
            #Group atoms by their chain index and filter unwanted residues indices (Effectively the data cleaning)
            toBeTransfAtomGroup,completeAtomGroup = self.groupAtomsByChain(atomGroup.select('stdaa'),filterList = unwantedResindices,bothResults=True)
            
            #BENCHMARK
            startTime = time.time()
            
            #Chains are being considered as standalone structures
         
            #================== Data Transformation==============================

            #For each chain
            for (toBeTransfchain,completeChain) in zip(toBeTransfAtomGroup,completeAtomGroup):
            
                #Store chain residues indices containing only the ones whose shall be transformed
                chainResIdx = self.atomGroupToResIdx(toBeTransfchain)
            
                #For each residue in chain
                for centerResIdx in chainResIdx:
                
                    #Data Transformation variables definitions
                    atomElements = []
                    atomCoordinates = []
                    alphaCarbonCoord = carbonCoord = oxigenCoord = nitrogenCoord = None
                    resName = None
            
                    #Get atom coordinates/elements data + remove centralized residue lateral chain atoms (We don't want to give our AI any clues of the residue :D)
                    atomElements,atomCoordinates,resName,alphaCarbonCoord,carbonCoord,oxigenCoord,nitrogenCoord = self.getElementsCoordsResname(completeChain,centerResIdx)
                    #DEBUG
                    #print('--------RES ',centerResIdx,'==>',resName,'----------')
                    
                    #Open storage file
                    f = self.startStorageFile(protName,resName,centerResIdx,storagePath)
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
                    fm = self.calculateNormalizedMatrix(carbonCoord,alphaCarbonCoord,oxigenCoord)
            
                    #Find rotation matrix. (As translation has already been done, translate = False)
                    rm = self.findRotationMatrix(im,fm,translate = False)
            
                    #Rotate
                    im_r = self.rotate(im,rm)
            
                    #Check rotation error
                    rmse = self.calcRmse(fm,im_r,3)
                    #DEBUG
                    #print('RMSE value should be near 0\n','RMSE:',rmse)
            
                    #BENCHMARK
                    endRotIdentifTime = time.time()
                    #DEBBUG
                    #print('Rotation Matrix Identification time: ',endRotIdentifTime - rotIdentifTime)
            
                    #BENCHMARK
                    rotTime = time.time()
            
                    #Rotate Structure based on centralized residue normalization
                    atomCoordinates = self.rotate(atomCoordinates,rm)
            
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
            
                    discGrid = np.zeros(shape=(self.n_elements,self.grid_dim,self.grid_dim,self.grid_dim),dtype=np.bool)
            
                    for coord,element in zip(atomCoordinates,atomElements):
            
                        r = self.calcGridPosition(coord[0,0],coord[0,1],coord[0,2],element,self.grid_scale,self.grid_dim)
            
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
                    self.storeGrid(f,discGrid)
            
                    #Close protein storage file
                    self.finishStorageFile(f)
            
            endTime = time.time()
        except Exception as e:

            print('Exception: ', e)
            #Close protein storage file
            if(f in locals()):
                self.finishStorageFile(f)
                errors.append(filename)
                #Delete file with bad data
                remove(storagePath + filename + '.hdf5')

        #DEBUG
        #print('Total time: ',endTime - startTime)
        
        #DEBUG
        #print('Voxel Coexistences: ',voxelCoexistence)
        
        print('Transformation Finished ==>  ' + filename)
        return voxelCoexistence , errors 