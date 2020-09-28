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

"""
NOTES: This is work in progress

* It takes ~2min using BETA_ITERATIONS=150
* It currently does not find meaningful solutions with 1000 reviews
* I don't understand alpha/mu/sigma initialization (dirichlet distribution and
  the paper does not specify)
* The paper says alpha is 0 <= a <= 1 but this is not the case here
* I don't understand how the (log)-likelihood is being calculated
"""

# reader = ReadData()
# reader.readAspectSeedWords()
# reader.readStopWords()
# reader.readReviewsFromJson()
# reader.removeLessFreqWords()
# bootstrap = BootStrap(reader)
# bootstrap.bootStrap()
# bootstrap.populateLists()

# assert len(bootstrap.wList) > 0, 'words list must not be empty'
# bootstrap.saveToFile("wList.json", bootstrap.wList)

# assert len(bootstrap.ratingsList) > 0, 'ratings list must not be empty'
# bootstrap.saveToFile("ratingsList.json", bootstrap.ratingsList)

# assert len(bootstrap.reviewIdList) > 0, 'ratings list must not be empty'
# bootstrap.saveToFile("reviewIdList.json", bootstrap.reviewIdList)
# bootstrap.saveToFile("vocab.json", list(bootstrap.corpus.wordFreq.keys()))
# bootstrap.saveToFile("aspectKeywords.json", bootstrap.corpus.aspectKeywords)

np.seterr(all='raise')
lrr_solver = LRR(
    should_assert=True
)
lrr_solver.solve(maxIter=20, covergence_threshold=0.0001)
lrr_solver.testing()
