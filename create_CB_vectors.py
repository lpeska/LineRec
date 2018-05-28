# -*- coding: utf-8 -*-
# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import math
import pandas as pd
import json
from scipy.spatial import distance

def uniArray(array_unicode):
    items = [str(x).encode('utf-8') for x in array_unicode]
    array_unicode = np.array([items]) # remove the brackets for line breaks
    return array_unicode

def getFeaturesFromUser(i, vec):
    featureList = []
    
    if i < 4653:
        featureList.append("GENDER M")
    else: 
        featureList.append("GENDER F")
    
    featureList.append("NATIONALITY "+str(vec.nationality[i]))
        
    if not math.isnan(float(vec.born[i])):
        featureList.append("BORN "+str(vec.born[i])) 
        age = 2017 - int(vec.born[i])

        if age < 18:
            featureList.append("age_0-18")
        if age > 15 and age < 25:
            featureList.append("age_15-25")
        if age > 20 and age < 30:
            featureList.append("age_20-30")
        if age > 25 and age < 35:
            featureList.append("age_25-35")   
        if age > 30 and age < 40:
            featureList.append("age_30-40") 
        if age > 35 and age < 45:
            featureList.append("age_35-45") 
        if age > 40 and age < 50:
            featureList.append("age_40-50")
        if age > 45 and age < 55:
            featureList.append("age_45-55")  
        if age > 50 and age < 60:
            featureList.append("age_50-60")
        if age > 55 and age < 65:
            featureList.append("age_55-65") 
        if age > 60 and age < 70:
            featureList.append("age_60-70")                 
        if age > 65:
            featureList.append("age_65-") 
    fl = vec.features[i].split(",")   
    featureList.extend(fl)    
    return featureList

if __name__ == "__main__":
    vec = pd.read_csv("sourceData/personsData.csv", sep=';', header=0)
    with open("sourceData/userIDs.csv", "r") as f:
        users = [int(line) for line in f]    
    featureList = []
    maxM = 4652
    #print(users)
    #find all possible features
    for i in range(0, len(vec)):#len(vec)): maxM):
        if int(vec.pid[i]) in  users:
            featureList.extend( getFeaturesFromUser(i, vec) )
    
    featureSet = set(featureList)  
    finalFeatureList = list(featureSet)
    print(len(featureSet), len(finalFeatureList))
    userMatrix = np.zeros((len(users), len(set(featureList))))
    
    print(userMatrix.shape)
    fotoIDList = []
    j = 0
    for i in range(0,  len(vec)):#len(vec)): maxM):
        try:
            if int(vec.pid[i]) in  users:
                userFeatures = getFeaturesFromUser(i, vec)
                indeces = [ finalFeatureList.index( feature )  for feature in userFeatures]
                userMatrix[j, indeces] = 1
                fotoIDList.append(str(vec.pid[i]))
                j = j+1
                #print(j)
        except:
            print("Error user:" + str(i))
            print(userFeatures)
            j = j+1
            print(j)
        
        
            
    sum_vector = np.sum(userMatrix, axis=0)  
    #print(finalFeatureList)
    np.savetxt("sourceData/finalFeatureList.csv", uniArray(finalFeatureList), delimiter=';', fmt="%s" )
    np.savetxt("testOutput/sum_vector.csv", sum_vector, delimiter=';')
    
    #idf_vector = [math.log(len(users)/count, 10) for count in sum_vector]
    #userMatrix = userMatrix * idf_vector
    

    """
    simMetric = distance.cdist(userMatrix, userMatrix, 'cosine') 
    simMetric = 1 - simMetric
    
    np.savetxt("cbDistance.csv", simMetric, delimiter=';')
    """
    #maxVal = np.amax(userMatrix)
    #userMatrix = userMatrix / maxVal
    
    np.savetxt("sourceData/personsCBVectors.csv", userMatrix, delimiter=';' , fmt="%i" )
    
    out_df = pd.DataFrame(userMatrix,  columns=finalFeatureList)
    out_df["id"] = users
    out_df.to_csv('sourceData/features.csv', index=False, header=True, sep=';', encoding="utf-8")    
    #np.savetxt("personsCB_IDs.csv", fotoIDList, delimiter=';', fmt='%s' )
    
    #ziskat hodnoty vlastnosti pro filtry
    listOfKeywords = ['BORN','GENDER','NATIONALITY','age_','BRADAVICE','TETOVÁNÍ','JIZVA','KOŽNÍ DEFEKT','PIGM. SKVRNA','POSTAVA','ZNALOST JAZYKŮ','BARVA VLASŮ','TVAR VLASŮ','BARVA VOUSŮ','STŘIH VOUSŮ','TVAR VOUSŮ','HUSTOTA VOUSŮ']
    featureValueMap = {}
    for j in range(0,  len(listOfKeywords)):
        featureValueMap[listOfKeywords[j]] = []
        
    for i in range(0,  len(finalFeatureList)):
        for j in range(0,  len(listOfKeywords)):
            if listOfKeywords[j] in str(finalFeatureList[i]) :
                featureValueMap[listOfKeywords[j]].append([i,finalFeatureList[i].replace(listOfKeywords[j],"")])
             
    
    with open('sourceData/featureVals.json', 'w') as fp:
        json.dump(featureValueMap, fp, sort_keys=True, indent=4)                
    print(featureValueMap)
    #print(userMatrix) 
   
    
    

