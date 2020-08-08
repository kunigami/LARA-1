import numpy as np
from scipy import optimize
import json
import random
import logging

modelDataDir = "modelData/"


class LRR:
    def __init__(self):
        words = self.loadDataFromFile("vocab.json")
        self.word_index_mapping = self.createWordIndexMapping(words)
        self.words_cnt = len(self.word_index_mapping)

        # Maps aspects to related keywords
        aspectKeywords = self.loadDataFromFile("aspectKeywords.json")
        self.aspect_index_mapping = self.createAspectIndexMapping(aspectKeywords)
        self.aspect_cnt = len(self.aspect_index_mapping)

        # Histogram of words for each review and aspect
        self.wList = self.loadDataFromFile("wList.json")

        # List of ratings for each aspect belonging to a review
        self.aspect_ratings = self.loadDataFromFile("ratingsList.json")

        # List of review IDs
        self.reviews_ids = self.loadDataFromFile("reviewIdList.json")
        self.reviews_cnt = len(self.reviews_ids)
        assert self.reviews_cnt > 0, "Reviews should exist in reviewIdList.json"

        # breaking dataset into 3:1 ratio, 3 parts for training and 1 for testing
        self.train_indices = random.sample(range(0, self.reviews_cnt), int(0.75 * self.reviews_cnt))
        self.test_indices = list(set(range(0, self.reviews_cnt)) - set(self.train_indices))

        self.train_reviews_cnt = len(self.train_indices)

        # delta - is simply a number
        self.delta = 1.0

        # matrix of aspect rating vectors (Sd) of all reviews - k*Rn
        self.S = np.empty(shape=(self.aspect_cnt, self.train_reviews_cnt), dtype=np.float64)

        # matrix of alphas (Alpha-d) of all reviews - k*Rn
        # each column represents Aplha-d vector for a review
        self.alpha = np.random.dirichlet(np.ones(self.aspect_cnt), size=1).reshape(self.aspect_cnt, 1)
        for i in range(self.train_reviews_cnt - 1):
            self.alpha = np.hstack(
                (
                    self.alpha,
                    np.random.dirichlet(np.ones(self.aspect_cnt), size=1).reshape(self.aspect_cnt, 1),
                )
            )

        # vector mu - k*1 vector
        self.mu = np.random.dirichlet(np.ones(self.aspect_cnt), size=1).reshape(self.aspect_cnt, 1)

        # matrix Beta for the whole corpus (for all aspects, for all words) - k*n matrix
        self.beta = np.random.uniform(low=-0.1, high=0.1, size=(self.aspect_cnt, self.words_cnt))

        # matrix sigma for the whole corpus - k*k matrix
        # Sigma needs to be positive definite, with diagonal elems positive
        """self.sigma = np.random.uniform(low=-1.0, high=1.0, size=(self.aspect_cnt, self.aspect_cnt))
        self.sigma = np.dot(self.sigma, self.sigma.transpose())
        print(self.sigma.shape, self.sigma)
        """

        # Following is help taken from:
        # https://stats.stackexchange.com/questions/124538/
        W = np.random.randn(self.aspect_cnt, self.aspect_cnt - 1)
        S = np.add(np.dot(W, W.transpose()), np.diag(np.random.rand(self.aspect_cnt)))
        D = np.diag(np.reciprocal(np.sqrt(np.diagonal(S))))
        self.sigma = np.dot(D, np.dot(S, D))
        self.sigmaInv = np.linalg.inv(self.sigma)

        """ testing for positive semi definite
        if(np.all(np.linalg.eigvals(self.sigma) > 0)): #whether is positive semi definite
            print("yes")
        print(self.sigma)
        """
        self.setup_logger()


    def setup_logger(self):
        self.logger = logging.getLogger("LRR")
        self.logger.setLevel(logging.INFO)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("output/lrr.log")
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def createWordIndexMapping(self, words):
        word_index_mapping = {}
        for i in range(len(words)):
            word_index_mapping[words[i]] = i
        return word_index_mapping

    def createAspectIndexMapping(self, aspect_keywords):
        aspect_index_mapping = {}
        aspects = list(aspect_keywords.keys())
        for i in range(len(aspects)):
            aspect_index_mapping[aspects[i]] = i
        return aspect_index_mapping

    def loadDataFromFile(self, fileName):
        json_data = None
        with open(modelDataDir + fileName, "r") as fp:
            json_data = json.load(fp)
        return json_data

    # given a dictionary as in every index of self.wList,
    # creates a W matrix as was in the paper
    def createWMatrix(self, w):
        W = np.zeros(shape=(self.aspect_cnt, self.words_cnt))
        for aspect, Dict in w.items():
            for word, freq in Dict.items():
                W[self.aspect_index_mapping[aspect]][self.word_index_mapping[word]] = freq
        return W

    # Computing aspectRating array for each review given Wd->W matrix for review 'd'
    def calcAspectRatings(self, Wd):
        Sd = np.einsum("ij,ij->i", self.beta, Wd).reshape((self.aspect_cnt,))
        try:
            Sd = np.exp(Sd)
        except Exception as inst:
            self.logger.info("Exception in calcAspectRatings : %s", Sd)
        return Sd

    def calcMu(self):  # calculates mu for (t+1)th iteration
        return np.sum(self.alpha, axis=1).reshape((self.aspect_cnt, 1)) / self.train_reviews_cnt

    def calcSigma(self, updateDiagonalsOnly):  # update diagonal entries only
        self.sigma.fill(0)
        for i in range(self.train_reviews_cnt):
            columnVec = self.alpha[:, i].reshape((self.aspect_cnt, 1))
            columnVec = columnVec - self.mu
            if updateDiagonalsOnly:
                for k in range(self.aspect_cnt):
                    self.sigma[k][k] += columnVec[k] * columnVec[k]
            else:
                self.sigma = self.sigma + np.dot(columnVec, columnVec.transpose())
        for i in range(self.aspect_cnt):
            self.sigma[i][i] = (1.0 + self.sigma[i][i]) / (1.0 + self.train_reviews_cnt)
        self.sigmaInv = np.linalg.inv(self.sigma)

    def calcOverallRating(self, alphaD, Sd):
        return np.dot(alphaD.transpose(), Sd)[0][0]

    def calcDeltaSquare(self):
        delta = 0.0
        for i in range(self.train_reviews_cnt):
            alphaD = self.alpha[:, i].reshape((self.aspect_cnt, 1))
            Sd = self.S[:, i].reshape((self.aspect_cnt, 1))
            Rd = float(self.aspect_ratings[self.train_indices[i]]["Overall"])
            temp = Rd - self.calcOverallRating(alphaD, Sd)
            try:
                delta += temp * temp
            except Exception:
                self.logger.info("Exception in Delta calc")
        return delta / self.train_reviews_cnt

    # TODO: this function is expensive. Can it be optimized?
    def maximumLikelihoodBeta(self, x, *args):
        beta = x
        beta = beta.reshape((self.aspect_cnt, self.words_cnt))
        innerBracket = np.empty(shape=self.train_reviews_cnt)
        for d in range(self.train_reviews_cnt):
            tmp = 0.0
            review_idx = self.train_indices[d]  # review index in wList
            W = self.createWMatrix(self.wList[review_idx])
            for i in range(self.aspect_cnt):
                tmp += (
                    self.alpha[i][d]
                    * np.dot(
                        beta[i, :].reshape((1, self.words_cnt)), W[i, :].reshape((self.words_cnt, 1))
                    )[0][0]
                )
            innerBracket[d] = tmp - float(self.aspect_ratings[review_idx]["Overall"])
        mlBeta = 0.0
        for d in range(self.train_reviews_cnt):
            mlBeta += innerBracket[d] * innerBracket[d]
        return mlBeta / (2 * self.delta)

    def gradBeta(self, x, *args):
        beta = x
        beta = beta.reshape((self.aspect_cnt, self.words_cnt))
        gradBetaMat = np.empty(shape=((self.aspect_cnt, self.words_cnt)), dtype="float64")
        innerBracket = np.empty(shape=self.train_reviews_cnt)
        for d in range(self.train_reviews_cnt):
            tmp = 0.0
            review_idx = self.train_indices[d]
            Wd = self.createWMatrix(self.wList[review_idx])
            for i in range(self.aspect_cnt):
                tmp += (
                    self.alpha[i][d]
                    * np.dot(
                        beta[i, :].reshape((1, self.words_cnt)), Wd[i, :].reshape((self.words_cnt, 1))
                    )[0][0]
                )
            innerBracket[d] = tmp - float(self.aspect_ratings[review_idx]["Overall"])

        for i in range(self.aspect_cnt):
            beta_i = np.zeros(shape=(1, self.words_cnt))
            for d in range(self.train_reviews_cnt):
                review_idx = self.train_indices[d]  # review index in wList
                W = self.createWMatrix(self.wList[review_idx])
                beta_i += innerBracket[d] * self.alpha[i][d] * W[i, :]
            gradBetaMat[i, :] = beta_i
        return gradBetaMat.reshape((self.aspect_cnt * self.words_cnt,))

    def calcBeta(self):
        beta, retVal, flags = optimize.fmin_l_bfgs_b(
            func=self.maximumLikelihoodBeta,
            x0=self.beta,
            fprime=self.gradBeta,
            args=(),
            m=5,
            maxiter=15,
        )
        converged = True
        if flags["warnflag"] != 0:
            converged = False
        self.logger.info("Beta converged? : %s", 'yes' if converged else 'no')
        return beta.reshape((self.aspect_cnt, self.words_cnt)), converged

    def maximumLikelihoodAlpha(self, x, *args):
        alphad = x
        alphad = alphad.reshape((self.aspect_cnt, 1))
        rd, Sd, deltasq, mu, sigmaInv = args
        temp1 = rd - np.dot(alphad.transpose(), Sd)[0][0]
        temp1 *= temp1
        temp1 /= deltasq * 2
        temp2 = alphad - mu
        temp2 = np.dot(np.dot(temp2.transpose(), sigmaInv), temp2)[0][0]
        temp2 /= 2
        return temp1 + temp2

    def gradAlpha(self, x, *args):
        alphad = x
        alphad = alphad.reshape((self.aspect_cnt, 1))
        rd, Sd, deltasq, mu, sigmaInv = args
        temp1 = (np.dot(alphad.transpose(), Sd)[0][0] - rd) * Sd
        temp1 /= deltasq
        temp2 = np.dot(sigmaInv, (alphad - mu))
        return (temp1 + temp2).reshape((self.aspect_cnt,))

    def calcAlphaD(self, i):
        alphaD = self.alpha[:, i].reshape((self.aspect_cnt, 1))
        review_idx = self.train_indices[i]
        rd = float(self.aspect_ratings[review_idx]["Overall"])
        Sd = self.S[:, i].reshape((self.aspect_cnt, 1))
        Args = (rd, Sd, self.delta, self.mu, self.sigmaInv)
        bounds = [(0, 1)] * self.aspect_cnt
        # self.gradf(alphaD, *Args)
        alphaD, retVal, flags = optimize.fmin_l_bfgs_b(
            func=self.maximumLikelihoodAlpha,
            x0=alphaD,
            fprime=self.gradAlpha,
            args=Args,
            bounds=bounds,
            m=5,
            maxiter=15,
        )
        converged = True
        if flags["warnflag"] != 0:
            converged = False
        self.logger.info("Alpha Converged : %d", flags["warnflag"])
        # Normalizing alphaD so that it follows dirichlet distribution
        alphaD = np.exp(alphaD)
        alphaD = alphaD / (np.sum(alphaD))
        return alphaD.reshape((self.aspect_cnt,)), converged

    """
    def getBetaLikelihood(self):
        likelihood=0
        return self.lambda*np.sum(np.einsum('ij,ij->i',self.beta,self.beta))
    """

    def dataLikelihood(self):
        likelihood = 0.0
        for d in range(self.train_reviews_cnt):
            review_idx = self.train_indices[d]
            Rd = float(self.aspect_ratings[review_idx]["Overall"])
            W = self.createWMatrix(self.wList[review_idx])
            Sd = self.calcAspectRatings(W).reshape((self.aspect_cnt, 1))
            alphaD = self.alpha[:, d].reshape((self.aspect_cnt, 1))
            temp = Rd - self.calcOverallRating(alphaD, Sd)
            try:
                likelihood += temp * temp
            except Exception:
                self.logger.debug("Exception in dataLikelihood")
        likelihood /= self.delta
        return likelihood

    def alphaLikelihood(self):
        likelihood = 0.0
        for d in range(self.train_reviews_cnt):
            alphad = self.alpha[:, d].reshape((self.aspect_cnt, 1))
            temp2 = alphad - self.mu
            temp2 = np.dot(np.dot(temp2.transpose(), self.sigmaInv), temp2)[0]
            likelihood += temp2
        try:
            likelihood += np.log(np.linalg.det(self.sigma))
        except FloatingPointError:
            self.logger.debug(
                "Exception in alphaLikelihood: %f", np.linalg.det(self.sigma)
            )
        return likelihood

    def calcLikelihood(self):
        likelihood = 0.0
        likelihood += np.log(self.delta)  # delta likelihood
        likelihood += (
            self.dataLikelihood()
        )  # data likelihood - will capture beta likelihood too
        likelihood += self.alphaLikelihood()  # alpha likelihood
        return likelihood

    def EStep(self):
        for i in range(self.train_reviews_cnt):
            review_idx = self.train_indices[i]
            W = self.createWMatrix(self.wList[review_idx])
            self.S[:, i] = self.calcAspectRatings(W)
            alphaD, converged = self.calcAlphaD(i)
            if converged:
                self.alpha[:, i] = alphaD
            self.logger.info("Alpha calculated")

    def MStep(self):
        likelihood = 0.0
        self.mu = self.calcMu()
        self.logger.info("Mu calculated")
        self.calcSigma(False)
        self.logger.info("Sigma calculated : %s " % np.linalg.det(self.sigma))
        likelihood += self.alphaLikelihood()  # alpha likelihood
        self.logger.info("alphaLikelihood calculated")
        print("alphaLikelihood calculated")
        beta, converged = self.calcBeta()
        if converged:
            self.beta = beta
        self.logger.info("Beta calculated")
        likelihood += (
            self.dataLikelihood()
        )  # data likelihood - will capture beta likelihood too
        self.logger.info("dataLikelihood calculated")

        self.delta = self.calcDeltaSquare()

        self.logger.info("Deltasq calculated")
        likelihood += np.log(self.delta)  # delta likelihood
        return likelihood

    def EMAlgo(self, maxIter, coverge):
        self.logger.info("Training started")
        iteration = 0
        old_likelihood = self.calcLikelihood()
        self.logger.info(
            "initial calcLikelihood calculated, det(Sig): %s"
            % np.linalg.det(self.sigma)
        )
        diff = 10.0
        while iteration < min(8, maxIter) or (iteration < maxIter and diff > coverge):
            print('Iteration', iteration)
            self.EStep()
            self.logger.info("EStep completed")
            print("EStep completed")
            likelihood = self.MStep()
            self.logger.info("MStep completed")
            print("MStep completed")
            diff = (old_likelihood - likelihood) / old_likelihood
            old_likelihood = likelihood
            iteration += 1
            self.logger.info("MStep completed %d (of %d)", iteration, maxIter)
        self.logger.info("Training completed")

    def testing(self):
        for i in range(self.reviews_cnt - self.train_reviews_cnt):
            review_idx = self.test_indices[i]
            W = self.createWMatrix(self.wList[review_idx])
            Sd = self.calcAspectRatings(W).reshape((self.aspect_cnt, 1))
            overallRating = self.calcOverallRating(self.mu, Sd)
            print("ReviewId-", self.reviews_ids[review_idx])
            print("Actual OverallRating:", self.aspect_ratings[review_idx]["Overall"])
            print("Predicted OverallRating:", overallRating)
            print("Actual vs Predicted Aspect Ratings:")
            for aspect, rating in self.aspect_ratings[review_idx].items():
                if (
                    aspect != "Overall"
                    and aspect.lower() in self.aspect_index_mapping.keys()
                ):
                    r = self.aspect_index_mapping[aspect.lower()]
                    print("Aspect:", aspect, " Rating:", rating, "Predic:", Sd[r])
            if overallRating > 3.0:
                print("Positive Review")
            else:
                print("Negative Review")
