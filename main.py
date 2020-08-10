from BootStrap import BootStrap
from LRR import LRR
from ReadData import ReadData

import os
import logging
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# This is where we store the temporary artifacts created
if not os.path.exists('output'):
    os.mkdir('output')


reader = ReadData()
reader.readAspectSeedWords()
reader.readStopWords()
reader.readReviewsFromJson()
reader.removeLessFreqWords()
bootstrap = BootStrap(reader)
bootstrap.bootStrap()
bootstrap.populateLists()

assert len(bootstrap.wList) > 0, 'words list must not be empty'
bootstrap.saveToFile("wList.json", bootstrap.wList)

assert len(bootstrap.ratingsList) > 0, 'ratings list must not be empty'
bootstrap.saveToFile("ratingsList.json", bootstrap.ratingsList)

assert len(bootstrap.reviewIdList) > 0, 'ratings list must not be empty'
bootstrap.saveToFile("reviewIdList.json", bootstrap.reviewIdList)

bootstrap.saveToFile("vocab.json", list(bootstrap.corpus.wordFreq.keys()))
bootstrap.saveToFile("aspectKeywords.json", bootstrap.corpus.aspectKeywords)

np.seterr(all='raise')
lrr_solver = LRR()
lrr_solver.EMAlgo(maxIter=2, coverge=0.0001)
lrr_solver.testing()
