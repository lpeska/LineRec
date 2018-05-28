# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

#lineupIDs[evaluatorID] = list of all performed lineups per evaluator
#selectedCandidates[evaluatorID_lineupID] = list of all candidates selected by the evaluator for particular lineup
#allCandidates[lineupID] = list of all candidates for particular lineup
#objectData[oid] = considered attributes of each object (CB, VIS or both) 
#output = uid - lineupID - recType - similarity - selected - features
import numpy as np
import pandas as pd
import math
import string
import itertools
from scipy.spatial import distance

def getAllPairs(list):
    return itertools.combinations(list,2)

def getKPairs(list1, list2, length):    
    el1 = np.random.choice(list1, length, replace=True)
    el2 = np.random.choice(list2, length, replace=True) 
    return zip(el1, el2)

def create_pairs(knownNegExamplesProb, unknownNegExamplesProb):   
    np.random.seed(1254)
    # knownNegExamplesProb + unknownNegExamplesProb + posExamplesProb = 1
    # define percentage of negative samples in the batch
    
    vv = pd.read_csv("sourceData/personsVectorsFC8.csv", sep=';', header=None)
    vec_vis = np.asarray(vv)[:,1:]
    
    vv2 = pd.read_csv("sourceData/personsVectorsFC8.csv", sep=';', header=None, dtype="str")
    fotoIDs = [str(f) for f in np.asarray(vv2)[:,0] ] 
    
    cb = pd.read_csv("sourceData/personsCBVectors.csv", sep=';', header=None)
    vec_cb = np.asarray(cb)
    
    pgs = pd.read_csv("sourceData/pageSummary.csv", sep=';', header=0, dtype="str")
    pg = np.asarray(pgs)
    
    results = pd.read_csv("sourceData/results.csv", sep=';', header=0, dtype="str")
    res = np.asarray(results)    
    
    with open("sourceData/userIDs.csv", "r") as f:
        users = [str(int(line)) for line in f]          
   
    origIDs = []    
    data = {}
    testPairs = []
    
    currLineupID = ""
    for i in range(0,pg.shape[0]):
        
        obj = pg[i,]
        fotoID = pg[i,0]
        fotoType = pg[i,1]
        
        fRow = fotoIDs.index(fotoID)

        if fotoType == "orig":
            origIDs.append(fotoID)
            currLineupID = fotoID
            data[currLineupID] = {"vis":[],"cb":[]}
        else:    
            #divide into vis and CB
            if fotoType == "vis":
                data[currLineupID]["vis"].append(fotoID)

            if fotoType == "cb":
                data[currLineupID]["cb"].append(fotoID)

    
    #with open("finalDatasetCB.csv", "w") as f:
    #    f.write("") 

    #create learning based on minimal FC8 distance for all examples (a sort of a baseline similarity - the more additional data we have, the more we can refine it
    defExamples = []
    dst = distance.cdist(vec_vis, vec_vis, 'cosine')
    for idx in range(len(fotoIDs)):
        currID = fotoIDs[idx]
        defPosCount = 10
        defNegCount = defPosCount * (unknownNegExamplesProb) / (1 - knownNegExamplesProb - unknownNegExamplesProb)
        simIDs = [fotoIDs[i] for i in dst[idx,:].argsort()[0:defPosCount]]
        defExamples.extend( [(currID,s,1) for s in simIDs] )
        nonSimIDs = [i for i in np.random.choice(fotoIDs, size=int(defNegCount), replace=True) ]
        defExamples.extend([(currID, s, 0) for s in nonSimIDs])
    testPairs.extend(defExamples)

    for j in range(0,res.shape[0]): 
        evaluator = res[j,1]
        lineup = res[j,3]
        selectedVis = res[j,8]
        selectedCB = res[j,9]   
        
        try:            
            sv = [int(k) for k in selectedVis.split(",")]
        except: 
            sv = []    
        try:      
            sc = [int(k) for k in selectedCB.split(",")] 
        except: 
            sc = []
        
        #sFID = foto IDs which were selected to the lineup, nFID = foto IDs which were proposed, but not selected, uFID = (sample of) foto ids which were not proposed
        sFID = []
        nFID = []
        uFID = []
        
        #print(evaluator, lineup, selectedVis, selectedCB)      
        for s in sc:
            sFID.append( data[lineup]["cb"][(s-1)] )
        for s in sv:
            sFID.append( data[lineup]["vis"][(s-1)] )
        sFID.append(str(lineup))    
        
        nFID = list( set(data[lineup]["cb"]).union(set(data[lineup]["vis"])) - set(sFID)  )
        
        uFID = list( set(users) - set(sFID).union(set(nFID)) )
        
        #print(sFID)
        #print(nFID)
        #print(uFID)
        
        posExamples = [(a,b,1) for (a,b) in list(getAllPairs(sFID))] 
        posLen = len(posExamples)
        posProb = 1 - knownNegExamplesProb - unknownNegExamplesProb #should be larger than 0
        negLen = int(round(posLen * knownNegExamplesProb / posProb ))
        unkLen = int(round(posLen * unknownNegExamplesProb / posProb ))


        cnnPosExamples = [(a, b, 1) for (a, b) in list(getKPairs(nFID, nFID, negLen))]  # capture original CNN similarity


        negExamples = [(a,b,0) for (a,b) in list(getKPairs( sFID , nFID, negLen))] #capture difference between similar&selected and similar items
        unkExamples = [(a,b,0) for (a,b) in list(getKPairs( list(set(sFID).union(set(nFID))) , uFID, unkLen))] #capture difference between similar and dissimilar items
        
        testPairs.extend(posExamples)
        testPairs.extend(cnnPosExamples)
        testPairs.extend(negExamples)
        testPairs.extend(unkExamples)
    return testPairs    

if __name__ == "__main__":
    
    tp = create_pairs(0.4, 0.5)
    np.savetxt("sourceData/allPairs.csv", tp, delimiter=';', fmt="%s" )
    
