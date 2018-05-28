""" Siamese implementation using Tensorflow with Foto Lineup examples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA

#import helpers
import inference
import visualize

contentMode = "cbvis"
visLayer = "FC7"

class LineupsData:
    pairs = []
    vecCB = []
    vecVis = []
    fids = []
    allDataLength = 0
    contentMode = "vis"
    visLayer = "FC7"
    
    def __init__(self, seed, contentMode, visLayer):
        self.visLayer = visLayer
        if self.visLayer == "PROB":
            self.visLayer = ""
        self.contentMode = contentMode

        (self.pairs, self.vecCB, self.vecVis, self.fids) = self.load_lineups_data()
        np.random.seed(seed)
        if(self.contentMode == "cb"):
            self.dtSize = self.vecCB.shape[1]
        elif(self.contentMode == "vis"):
            self.dtSize = self.vecVis.shape[1]
        else:
            self.dtSizeCB =  self.vecCB.shape[1]
            self.dtSizeVis = self.vecVis.shape[1] 

         
    def load_lineups_data(self):#return pairs, CBvectors, VisVectors, listFIDs
        vv = pd.read_csv("sourceData/personsVectors"+self.visLayer+".csv", sep=';', header=None)
        vec_vis = np.asarray(vv)[:,1:]

        vv2 = pd.read_csv("sourceData/personsVectors"+self.visLayer+".csv", sep=';', header=None, dtype="str")
        fids = [str(f) for f in np.asarray(vv2)[:,0] ] 
        self.allDataLength = len(fids)
        cb = pd.read_csv("sourceData/personsCBVectors.csv", sep=';', header=None)
        vec_cb = np.asarray(cb)

        pgs = pd.read_csv("sourceData/allPairs.csv", sep=';', header=0, dtype="str")
        pairs = np.asarray(pgs)

        return (pairs, vec_cb, vec_vis, fids)

    def next_batch(self, length):
        #batchIDs = np.random.choice(self.pairs, length)        
        batchIDs = self.pairs[np.random.choice(self.pairs.shape[0], length, replace=False), :]
        if(self.contentMode == "cb"):
            batch_x1=[self.vecCB[self.fids.index(a),:] for (a,b,c) in batchIDs ]
            batch_x2=[self.vecCB[self.fids.index(b),:] for (a,b,c) in batchIDs ]
            batch_y=[c for (a,b,c) in batchIDs ]
            return(batch_x1, batch_x2, batch_y)
        elif(self.contentMode == "vis"):
            batch_x1=[self.vecVis[self.fids.index(a),:] for (a,b,c) in batchIDs ]
            batch_x2=[self.vecVis[self.fids.index(b),:] for (a,b,c) in batchIDs ]
            batch_y=[c for (a,b,c) in batchIDs ]
            return(batch_x1, batch_x2, batch_y)
        else:
            batch_x11=[self.vecCB[self.fids.index(a),:] for (a,b,c) in batchIDs ]
            batch_x12=[self.vecVis[self.fids.index(a),:] for (a,b,c) in batchIDs ]
            
            batch_x21=[self.vecCB[self.fids.index(b),:] for (a,b,c) in batchIDs ]
            batch_x22=[self.vecVis[self.fids.index(b),:] for (a,b,c) in batchIDs ]
            
            batch_y=[c for (a,b,c) in batchIDs ]
            return(batch_x11, batch_x12, batch_x21, batch_x22, batch_y)
        
        
    def test(self, length):
        #batchIDs = np.random.choice(self.pairs, length)        
        fotoIDs = range(length)

        if(self.contentMode == "cb"):
            test_x1=[self.vecCB[a,:] for a in fotoIDs ]
            test_id=[self.fids[a] for a in fotoIDs ]
            return(test_x1, test_id)
        elif(self.contentMode == "vis"):
            test_x1=[self.vecVis[a,:] for a in fotoIDs ]
            test_id=[self.fids[a] for a in fotoIDs ]
            return(test_x1, test_id)
        else:
            test_x1=[self.vecCB[a,:] for a in fotoIDs ]
            test_x2=[self.vecVis[a,:] for a in fotoIDs ]
            test_id=[self.fids[a] for a in fotoIDs ]
            return(test_x1,test_x2, test_id)
        
    
lnData = LineupsData(2548, contentMode, visLayer)

print(lnData.vecCB.shape)
print(lnData.vecVis.shape)
print(lnData.pairs.shape)
print(len(lnData.fids))


import os
import os.path as path

# prepare log directories
dirname = ''
train_dirname = path.join('log', dirname + '-train')

os.makedirs(train_dirname)
print('Summaries will be written to  {} '.format(train_dirname))



# prepare data and tf.session
sess = tf.InteractiveSession()


# setup siamese network
if(lnData.contentMode == "cbvis"):
    siamese = inference.SiameseDual(lnData.contentMode, lnData.dtSizeCB, lnData.dtSizeVis)
else:
    siamese = inference.Siamese(lnData.contentMode, lnData.dtSize)

train_step = tf.train.AdamOptimizer(0.01).minimize(siamese.loss) #for FC7 0.002


tf.summary.scalar('loss', siamese.loss)
train_summaries = tf.summary.merge_all()
print('Writing graph')
train_writer = tf.summary.FileWriter(train_dirname, sess.graph)

saver = tf.train.Saver()
tf.initialize_all_variables().run()

# if you just want to load a previously trainmodel?
new = True
model_ckpt = os.path.join(os.getcwd(), "processData/"+contentMode+'_model.ckpt')
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = "no"#raw_input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False

# start training
step = 0
if new:
    for step in range(10001):#50001 for FC8 and PROB
        if(lnData.contentMode == "cbvis"):
            (batch_x11, batch_x12, batch_x21, batch_x22, batch_y) = lnData.next_batch(128)
            #print(batch_x1, batch_x2, batch_y)

            _, loss_v, tsum = sess.run([train_step, siamese.loss, train_summaries], feed_dict={
                                siamese.x11: batch_x11, 
                                siamese.x12: batch_x12, 
                                siamese.x21: batch_x21, 
                                siamese.x22: batch_x22, 
                                siamese.y_: batch_y})

            train_writer.add_summary(tsum, step)

        else:
            (batch_x1, batch_x2, batch_y) = lnData.next_batch(128)
            #print(batch_x1, batch_x2, batch_y)

            _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                                siamese.x1: batch_x1, 
                                siamese.x2: batch_x2, 
                                siamese.y_: batch_y})

        step += 1

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if step % 100 == 0:
            print ('step %d: loss %.3f' % (step, loss_v))

        if step % 1000 == 0 and step > 0:
            saver.save(sess, os.path.join(os.getcwd(), "processData/"+contentMode+visLayer+'_model.ckpt'))
            
            if(lnData.contentMode == "cbvis"):
                (testInput1,testInput2, testIDs) = lnData.test(lnData.allDataLength)
                embed = siamese.o1.eval({siamese.x11: testInput1,siamese.x12: testInput2})
            else:
                (testInput, testIDs) = lnData.test(lnData.allDataLength)
                embed = siamese.o1.eval({siamese.x1: testInput})
                
            np.savetxt("processData/"+contentMode+visLayer+"_test_embeddings.csv", embed, delimiter=";", fmt="%.8f")
            np.savetxt("processData/"+contentMode+visLayer+"_test_ids.csv", testIDs, delimiter=";", fmt="%s")
           
else:
    saver.restore(sess, os.path.join(os.getcwd(), "processData/"+contentMode+visLayer+'_model.ckpt'))

# visualize result
if(lnData.contentMode == "cbvis"):
    (testInput1,testInput2, testIDs) = lnData.test(50)
    embed = siamese.o1.eval({siamese.x11: testInput1,siamese.x12: testInput2})
else:
    (testInput, testIDs) = lnData.test(50)
    embed = siamese.o1.eval({siamese.x1: testInput}) 
    


pca = PCA(n_components=2)
pca.fit(embed)
embed2D = pca.transform(embed)
visualize.Visualize(embed2D, testIDs)