import numpy as np
import pandas as pd
import time
from scipy.spatial import distance

contentMode = "cbvis"
visLayers = ["FC8","FC7","PROB"] #"FC7","PROB"
requestedRecommendations = 7
suspectToCandidateImportanceRatio = 0.25

dst = np.zeros((4652,4652))
for visLayer in visLayers:
    vec = pd.read_csv("processData/"+contentMode+visLayer+"_test_embeddings.csv", sep=';', header=None)
    fids = pd.read_csv("processData/"+contentMode+visLayer+"_test_ids.csv", sep=";", header=None, dtype="str")
    fidList = fids[0].tolist()
    v = np.asarray(vec)
    print(v.shape)
    partDst = distance.cdist(v, v, 'cosine')
    partDst[partDst < 1e-14] = 1.0 # deal with identical objects
    dst = dst + partDst[0:4652, 0:4652]
print(dst.shape)

lineupIDs = ["8061205130408", "8100303170706", "10301206130703", "13430524130707", "8121007141903", "8300518161902", "8510203150012", "10150425165030"]
for lid in lineupIDs:
    t1 = time.time()
    print(t1)
    idx = fids.index[fids[0] == lid].tolist()[0]
    arr = np.repeat(dst[idx,:],1)
    recsSim = []
    recsIDs = []
    recsI = [idx]
    for rID in range(requestedRecommendations):
        recIdx = arr.argsort()[0:50]
        for i in recIdx:
            if (min(dst[recsI,i]) < 1e-14) or (max(dst[recsI,i]) > 0.999):
                print(min(dst[recsI,i]), max(dst[recsI,i]))
                continue #items same as already found ones
            recsSim.append(arr[i])
            recsIDs.append(fidList[i])
            recsI.append(i)
            addedDST = dst[i,:] * suspectToCandidateImportanceRatio
            arr = arr + addedDST
            break

    t2 = time.time()
    print(t2)
    print(t2 - t1)

    print(idx)
    print(recsI)
    print(recsSim)
    print(recsIDs)


    row = "cnn_"+contentMode+"COMBINED_UNIFORMITY;"+lid+";;"+",".join(recsIDs)+"\n"
    print(row)
    with open("testOutput/cnnResults.csv", "a") as fp:
        fp.write(row)

#np.savetxt("processData/"+contentMode+visLayer+"_cosineDistance.csv", dst, delimiter=';')



