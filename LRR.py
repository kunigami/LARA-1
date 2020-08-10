import numpy as np
from scipy import optimize
import json
import random
import logging

modelDataDir = "modelData/"

BETA_ITERATIONS = 15
# For floating point comparison
EPS = 0.0001

class LRR:
    def __init__(self, should_assert=False):
        self.should_assert = should_assert

        words = self.loadDataFromFile("vocab.json")
        self.word_index_mapping = self.createWordIndexMapping(words)
        self.words_cnt = len(self.word_index_mapping)

        # Maps aspects to related keywords
        aspectKeywords = self.loadDataFromFile("aspectKeywords.json")
        self.aspect_index_mapping = self.createAspectIndexMapping(aspectKeywords)
        self.aspect_cnt = len(self.aspect_index_mapping)

        # Histogram of words for each review and aspect
        self.wList = self.loadDataFromFile("wList.json")

        # List of ratings for each aspect belonging to a review ([reviews X aspects])
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
        self.delta_sq = 1.0

        # matrix of aspect rating vectors (Sd) of all reviews - [aspects X reviews]
        self.S = np.empty(shape=(self.aspect_cnt, self.train_reviews_cnt), dtype=np.float64)

        # matrix of alphas (Alpha-d) of all reviews - [reviews X aspects]
        # each column represents Aplha-d vector for a review
        self.alpha = np.random.dirichlet(np.ones(self.aspect_cnt), size=1).reshape(self.aspect_cnt, 1)
        for d in range(self.train_reviews_cnt - 1):
            self.alpha = np.hstack(
                (
                    self.alpha,
                    np.random.dirichlet(np.ones(self.aspect_cnt), size=1).reshape(self.aspect_cnt, 1),
                )
            )

        if self.should_assert:
            assert_alpha(self.alpha)

        # mean parameter for the Gaussian distribution for the aspect weights alpha (aspects X 1)
        self.mu = np.random.dirichlet(np.ones(self.aspect_cnt), size=1).reshape(self.aspect_cnt, 1)

        # matrix Beta for the whole corpus (for all aspects, for all words) - k*n matrix
        self.beta = np.random.uniform(low=-0.1, high=0.1, size=(self.aspect_cnt, self.words_cnt))

        self.Wd = []
        for d in range(self.reviews_cnt):
            self.Wd.append(self.createWMatrix(self.wList[d]))
        if should_assert:
            assert_words_matrix(self.Wd, reviews_cnt=self.reviews_cnt, aspect_cnt=self.aspect_cnt)

        # matrix sigma for the whole corpus - k*k matrix
        # Sigma needs to be positive definite, with diagonal elems positive
        """self.delta_sqigma = np.random.uniform(low=-1.0, high=1.0, size=(self.aspect_cnt, self.aspect_cnt))
        self.sigma = np.dot(self.sigma, self.sigma.transpose())
        print(self.sigma.shape, self.sigma)
        """

        # sigma - variance parameter for the Gaussian distribution (k x k)

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
        fh = logging.FileHandler("output/lrr.log", mode='w')
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
    #
    # creates a W matrix as was in the paper
    def createWMatrix(self, histo_by_aspect):
        W = np.zeros(shape=(self.aspect_cnt, self.words_cnt))
        for aspect, cnt_by_words in histo_by_aspect.items():
            total_count = sum(cnt_by_words.values())
            for word, cnt in cnt_by_words.items():
                aspect_index = self.aspect_index_mapping[aspect]
                word_index = self.word_index_mapping[word]
                W[aspect_index][word_index] = 1.0 * cnt / total_count
        return W

    # Computing aspectRating array for each review given Wd->W matrix for review 'd'
    def calc_aspect_ratings(self, Wd):
        # Inner product over words
        Sd = np.einsum("ij,ij->i", self.beta, Wd).reshape((self.aspect_cnt,))
        try:
            Sd = np.exp(Sd)
        except Exception as inst:
            self.logger.info("Exception in calc_aspect_ratings : %s", Sd)
        return Sd

    # calculates mu for (t+1)th iteration. Eq. 8 in the paper.
    def calc_mu(self):
        return np.sum(self.alpha, axis=1).reshape((self.aspect_cnt, 1)) / self.train_reviews_cnt

    # calculates sigma for (t+1)th iteration. Eq. 9 in the paper.
    def calc_sigma(self):
        self.sigma.fill(0)
        for d in range(self.train_reviews_cnt):
            alpha_aspects = self.alpha[:, d].reshape((self.aspect_cnt, 1))
            alpha_aspects = alpha_aspects - self.mu
            self.sigma = self.sigma + np.dot(alpha_aspects, alpha_aspects.transpose())

        for k in range(self.aspect_cnt):
            self.sigma[k][k] = (1.0 + self.sigma[k][k]) / (1.0 + self.train_reviews_cnt)

        self.sigmaInv = np.linalg.inv(self.sigma)

    def calc_overall_rating(self, alpha_d, Sd):
        return np.einsum('i,i->', alpha_d, Sd)

    # calculates delta square for (t+1)th iteration. Eq. 10 in the paper.
    def calc_delta_square(self):
        delta = 0.0
        for d in range(self.train_reviews_cnt):
            rd = float(self.aspect_ratings[self.train_indices[d]]["Overall"])

            alpha_d = self.alpha[:, d].reshape((self.aspect_cnt,))
            Sd = self.S[:, d].reshape((self.aspect_cnt,))
            delta += (rd - self.calc_overall_rating(alpha_d, Sd))**2
        return delta / self.train_reviews_cnt

    def maximumLikelihoodBeta(self, x, *args):
        return maximum_likelihood_beta(
            alpha=self.alpha,
            beta=x,
            delta=self.delta_sq,
            aspect_ratings=self.aspect_ratings,
            train_indices=self.train_indices,
            Wd=self.Wd,
            words_cnt=self.words_cnt
        )

    def gradBeta(self, x, *args):
        return maximum_likelihood_beta_grad(
            alpha=self.alpha,
            beta=x,
            aspect_ratings=self.aspect_ratings,
            train_indices=self.train_indices,
            Wd=self.Wd,
            words_cnt=self.words_cnt
        )

        grad_beta_mat = np.empty(shape=((self.aspect_cnt, self.words_cnt)), dtype="float64")
        inner_bracket = np.empty(shape=self.train_reviews_cnt)
        for d in range(self.train_reviews_cnt):
            tmp = 0.0
            review_idx = self.train_indices[d]
            Wd = self.Wd[review_idx]
            for i in range(self.aspect_cnt):
                tmp += (
                    self.alpha[i][d]
                    * np.dot(
                        beta[i, :].reshape((1, self.words_cnt)), Wd[i, :].reshape((self.words_cnt, 1))
                    )[0][0]
                )
            inner_bracket[d] = tmp - float(self.aspect_ratings[review_idx]["Overall"])

        for i in range(self.aspect_cnt):
            beta_i = np.zeros(shape=(1, self.words_cnt))
            for d in range(self.train_reviews_cnt):
                review_idx = self.train_indices[d]  # review index in wList
                W = self.Wd[review_idx]
                beta_i += inner_bracket[d] * self.alpha[i][d] * W[i, :]
            grad_beta_mat[i, :] = beta_i
        return grad_beta_mat.reshape((self.aspect_cnt * self.words_cnt,))

    def calcBeta(self):
        beta, retVal, flags = optimize.fmin_l_bfgs_b(
            func=self.maximumLikelihoodBeta,
            x0=self.beta,
            fprime=self.gradBeta,
            args=(),
            m=5,
            maxiter=BETA_ITERATIONS,
        )
        converged = True
        if flags["warnflag"] != 0:
            converged = False
        self.logger.info("Beta converged? : %s", 'yes' if converged else 'no')
        return beta.reshape((self.aspect_cnt, self.words_cnt)), converged

    def maximumLikelihoodAlpha(self, x, *args):
        alpha_d = x
        alpha_d = alpha_d.reshape((self.aspect_cnt, 1))
        rd, Sd, deltasq, mu, sigmaInv = args
        temp1 = rd - np.dot(alpha_d.transpose(), Sd)[0][0]
        temp1 *= temp1
        temp1 /= deltasq * 2
        temp2 = alpha_d - mu
        temp2 = np.dot(np.dot(temp2.transpose(), sigmaInv), temp2)[0][0]
        temp2 /= 2
        return temp1 + temp2

    def gradAlpha(self, x, *args):
        alpha_d = x
        alpha_d = alpha_d.reshape((self.aspect_cnt, 1))
        rd, Sd, deltasq, mu, sigmaInv = args
        temp1 = (np.dot(alpha_d.transpose(), Sd)[0][0] - rd) * Sd
        temp1 /= deltasq
        temp2 = np.dot(sigmaInv, (alpha_d - mu))
        return (temp1 + temp2).reshape((self.aspect_cnt,))

    def calc_alpha_d(self, i):
        alpha_d = self.alpha[:, i].reshape((self.aspect_cnt, 1))
        review_idx = self.train_indices[i]
        rd = float(self.aspect_ratings[review_idx]["Overall"])
        Sd = self.S[:, i].reshape((self.aspect_cnt, 1))
        Args = (rd, Sd, self.delta_sq, self.mu, self.sigmaInv)
        bounds = [(0, 1)] * self.aspect_cnt

        alpha_d, retVal, flags = optimize.fmin_l_bfgs_b(
            func=self.maximumLikelihoodAlpha,
            x0=alpha_d,
            fprime=self.gradAlpha,
            args=Args,
            bounds=bounds,
            m=5,
            maxiter=1500,
        )
        converged = True
        if flags["warnflag"] != 0:
            converged = False

        # self.logger.info("Alpha converged? : %s", 'yes' if converged else 'no')
        # Normalizing alpha_d so that it follows dirichlet distribution
        alpha_d = np.exp(alpha_d)
        alpha_d = alpha_d / (np.sum(alpha_d))

        return alpha_d.reshape((self.aspect_cnt,)), converged

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
            W = self.Wd[review_idx]
            Sd = self.calc_aspect_ratings(W).reshape((self.aspect_cnt,))
            alpha_d = self.alpha[:, d].reshape((self.aspect_cnt,))
            temp = Rd - self.calc_overall_rating(alpha_d, Sd)
            try:
                likelihood += temp * temp
            except Exception:
                self.logger.debug("Exception in dataLikelihood")
        likelihood /= self.delta_sq
        return likelihood

    def alpha_likelihood(self):
        likelihood = 0.0
        for d in range(self.train_reviews_cnt):
            alpha_d = self.alpha[:, d].reshape((self.aspect_cnt, 1))
            temp2 = alpha_d - self.mu
            temp2 = np.dot(np.dot(temp2.transpose(), self.sigmaInv), temp2)[0]
            likelihood += temp2
        try:
            likelihood += np.log(np.linalg.det(self.sigma))
        except FloatingPointError:
            self.logger.debug(
                "Exception in alpha_likelihood: %f", np.linalg.det(self.sigma)
            )
        return likelihood

    def calc_likelihood(self):
        return \
            np.log(self.delta_sq) + \
            self.dataLikelihood() + \
            self.alpha_likelihood()

    # Expectation calculation step
    def EStep(self):
        for d in range(self.train_reviews_cnt):
            review_idx = self.train_indices[d]
            W = self.Wd[review_idx]
            self.S[:, d] = self.calc_aspect_ratings(W)

            alpha_d, converged = self.calc_alpha_d(d)
            if converged:
                self.alpha[:, d] = alpha_d

        if self.should_assert:
            assert_alpha(self.alpha)

    # Maximization step
    def MStep(self):
        self.mu = self.calc_mu()
        self.logger.info("Mu calculated")

        self.calc_sigma()
        self.logger.info("Sigma calculated : %s " % np.linalg.det(self.sigma))

        self.logger.info("alpha_likelihood calculated")

        beta, converged = self.calcBeta()
        if converged:
            self.beta = beta
        self.logger.info("Beta calculated")
        self.logger.info("dataLikelihood calculated")

        self.delta_sq = self.calc_delta_square()
        self.logger.info("Deltasq calculated")

    def EMAlgo(self, maxIter, covergence_threshold):
        self.logger.info("Training started")
        iteration = 0
        old_likelihood = self.calc_likelihood()

        diff = covergence_threshold + 1
        while iteration < min(8, maxIter) or (iteration < maxIter and diff > covergence_threshold):
            self.EStep()
            self.logger.info("EStep completed")

            self.MStep()

            likelihood = self.calc_likelihood()

            self.logger.info("MStep completed")

            diff = (old_likelihood - likelihood) / old_likelihood
            old_likelihood = likelihood
            iteration += 1

            print("likelihood improvement: ", likelihood, diff)
            self.logger.info("MStep completed %d (of %d)", iteration, maxIter)

        self.logger.info("Training completed")

    def testing(self):
        mu = self.mu.reshape((self.aspect_cnt,))
        for i in range(10):
            # self.reviews_cnt - self.train_reviews_cnt):
            review_idx = self.test_indices[i]
            W = self.Wd[review_idx]
            Sd = self.calc_aspect_ratings(W).reshape((self.aspect_cnt,))
            overall_rating = self.calc_overall_rating(mu, Sd)
            print("Review:", self.reviews_ids[review_idx])
            print("Actual Rating:", self.aspect_ratings[review_idx]["Overall"])
            print("Predicted Rating:", overall_rating*5)
            print("Actual vs Predicted Aspect Ratings:")
            for aspect, rating in self.aspect_ratings[review_idx].items():
                if (
                    aspect != "Overall"
                    and aspect.lower() in self.aspect_index_mapping.keys()
                ):
                    r = self.aspect_index_mapping[aspect.lower()]
                    print("  Aspect: %15s | Rating: %s | Predicted: %.1f" % (aspect, rating, Sd[r]*5))
            if overall_rating > 3.0:
                print("Positive Review")
            else:
                print("Negative Review")
            print("")


def maximum_likelihood_beta(
    alpha, # [aspect X words]
    beta, # [aspect X words]
    delta, # scalar
    aspect_ratings, # [reviews]
    train_indices, # [reviews]
    Wd, # [reviews X aspect X words]
    words_cnt,
):
    aspect_cnt = len(alpha)
    train_reviews_cnt = len(train_indices)

    beta = beta.reshape((aspect_cnt, words_cnt))
    inner_bracket = np.empty(shape=train_reviews_cnt)

    for d in range(train_reviews_cnt):
        review_idx = train_indices[d]  # review index in wList
        W = Wd[review_idx]

        inner_prods = np.einsum("ij,ij->i", beta, W)
        tmp =  np.einsum("i,i->", alpha[:, d], inner_prods)

        inner_bracket[d] = tmp - float(aspect_ratings[review_idx]["Overall"])

    ml_beta = np.einsum("i,i->", inner_bracket, inner_bracket)
    return ml_beta / (2 * delta)

def maximum_likelihood_beta_grad(
    alpha, # [aspect X words]
    beta, # [aspect X words]
    aspect_ratings, # [reviews]
    train_indices, # [reviews]
    Wd, # [reviews X aspect X words]
    words_cnt,
): # [aspect X words]
    aspect_cnt = len(alpha)
    train_reviews_cnt = len(train_indices)
    beta = beta.reshape((aspect_cnt, words_cnt))

    inner_bracket = np.empty(shape=train_reviews_cnt)
    for d in range(train_reviews_cnt):
        tmp = 0.0
        review_idx = train_indices[d]
        W = Wd[review_idx]

        inner_prods = np.einsum("ij,ij->i", beta, W)
        tmp =  np.einsum("i,i->", alpha[:, d], inner_prods)
        inner_bracket[d] = tmp - float(aspect_ratings[review_idx]["Overall"])

    grad_beta_mat = np.empty(shape=((aspect_cnt, words_cnt)), dtype="float64")
    for i in range(aspect_cnt):
        beta_i = np.zeros(shape=(1, words_cnt))
        for d in range(train_reviews_cnt):
            review_idx = train_indices[d]  # review index in wList
            W = Wd[review_idx]
            beta_i += inner_bracket[d] * alpha[i][d] * W[i, :]
        grad_beta_mat[i, :] = beta_i
    return grad_beta_mat.reshape((aspect_cnt * words_cnt,))

def assert_words_matrix(Wd, reviews_cnt, aspect_cnt):
    assert reviews_cnt == len(Wd), "Wd's first dimension is the set of reviews"

    for d in range(reviews_cnt):
        freq_by_aspect = Wd[d]
        assert aspect_cnt == len(freq_by_aspect), \
            "freq_by_aspect first dimension is the set of aspects"

        for freq in freq_by_aspect:
            total = sum(freq)
            assert is_almost(total, 1.0) or is_almost(total, 0.0), "freq_by_aspect should be normalized for each aspect"

def assert_alpha(alpha):
    reviews_cnt = len(alpha)
    for d in range(reviews_cnt):
        total = 0.0
        for a in alpha[d]:
            # assert geq(a, 0.0) and leq(a, 1.0), "0 <= alpha[d][k] <= 1"
            total += a
        # assert is_almost(total, 1.0), "alpha[d] must add up to 1"


def is_almost(a, b):
    return abs(a - b) < EPS

def geq(a, b):
    return a > b - EPS

def leq(a, b):
    return a < b + EPS
