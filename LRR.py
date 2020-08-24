import numpy as np
from scipy import optimize
import json
import random
import logging

modelDataDir = "modelData/"

# BETA_ITERATIONS = 150000
# For floating point comparison
EPS = 0.0001

BETA_ITERATIONS = 500
ALPHA_ITERATIONS = 1500
# For floating point comparison
LAMBDA = 2 # regularization parameter for beta
BETA_FCTR = 1e12
ALPHA_FCTR = 1e12
PI = 0.5
# Typical values for factr are:
# 1e12 for low accuracy; 1e7 for moderate accuracy; 10.0 for extremely high accuracy.

class LRR():
    def __init__(self, should_assert=False):
        random.seed(0)

        self.lambda_param = LAMBDA
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
        # delta_sq should be the variance of normal dist rd is draw from
        # represent uncertainty of overall rating predictions
        self.delta_sq = 1.0

        # predefined confidence parameter to control L_aux influence
        # pi should be much smaller than 1/delta_sq
        self.pi = PI

        # matrix of aspect rating vectors (Sd) of all reviews - [aspects X reviews]
        # Aspect weight vector
        self.S = np.empty(shape=(self.aspect_cnt, self.train_reviews_cnt), dtype=np.float64)
        print(self.S)

        # mean parameter for the Gaussian distribution for the aspect weights alpha (aspects X 1)
        self.mu = np.random.dirichlet(np.ones(self.aspect_cnt), size=1).reshape(self.aspect_cnt, 1)
        self.mu = np.random.uniform(low=-1., high=1., size=(self.aspect_cnt, 1))

        # matrix Beta for the whole corpus (for all aspects, for all words) - k*n matrix
        # beta is word sentiment polarity on that aspect
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
        self.sigma = np.eye(self.aspect_cnt)
        self.sigmaInv = np.linalg.inv(self.sigma)


                # matrix of alphas (Alpha-d) of all reviews - [reviews X aspects]
        # each column represents Aplha-d aspect rating vector for a review
        self.alpha_hat = np.random.multivariate_normal(mean=self.mu.reshape(self.aspect_cnt, ), cov=self.sigma, size=1).reshape(self.aspect_cnt, 1)
        self.alpha = np.exp(self.alpha_hat) / (np.sum(np.exp(self.alpha_hat)))

        for d in range(self.train_reviews_cnt - 1):
            alpha_hat_add = np.random.multivariate_normal(mean=self.mu.reshape(self.aspect_cnt, ), cov=self.sigma, size=1).reshape(self.aspect_cnt, 1)
            self.alpha_hat = np.hstack(
                (
                    self.alpha_hat,
                    alpha_hat_add,
                )
            )

            self.alpha = np.hstack(
                (
                    self.alpha,
                    (np.exp(alpha_hat_add) / np.sum(np.exp(alpha_hat_add))).reshape(self.aspect_cnt, 1)
                )
            )

#         self.alpha_hat = np.random.dirichlet(np.ones(self.aspect_cnt), size=1).reshape(self.aspect_cnt, 1)
#         self.alpha = np.exp(self.alpha_hat) / (np.sum(np.exp(self.alpha_hat)))

#         for d in range(self.train_reviews_cnt - 1):
#             self.alpha_hat = np.hstack(
#                 (
#                     self.alpha_hat,
#                     np.random.dirichlet(np.ones(self.aspect_cnt), size=1).reshape(self.aspect_cnt, 1)
#                 )
#             )

#             self.alpha = np.hstack(
#                 (
#                     self.alpha,
#                     (np.exp(self.alpha_hat[:, d+1]) / np.sum(np.exp(self.alpha_hat[:, d+1]))).reshape(self.aspect_cnt, 1)
#                      ,
#                 )
#             )

        print('initial alphah', self.alpha_hat)

        if self.should_assert:
            assert_alpha(self.alpha)

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
        Sd = np.exp(Sd)
        print(Sd)
        return Sd

    # calculates mu for (t+1)th iteration. Eq. 8 in the paper.
    def calc_mu(self):
        # FIXME
        return np.sum(self.alpha_hat, axis=1).reshape((self.aspect_cnt, 1)) / self.train_reviews_cnt
        # return np.sum(self.alpha, axis=1).reshape((self.aspect_cnt, 1)) / self.train_reviews_cnt

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
            delta_sq=self.delta_sq,
            pi=self.pi,
            lambda_param = self.lambda_param,
            aspect_ratings=self.aspect_ratings,
            train_indices=self.train_indices,
            Wd=self.Wd,
            S=self.S,
            words_cnt=self.words_cnt
        )

    def gradBeta(self, x, *args):
        return maximum_likelihood_beta_grad(
            alpha=self.alpha,
            beta=x,
            delta_sq=self.delta_sq,
            pi=self.pi,
            lambda_param = self.lambda_param,
            aspect_ratings=self.aspect_ratings,
            train_indices=self.train_indices,
            Wd=self.Wd,
            S=self.S,
            words_cnt=self.words_cnt
        )

    def calcBeta(self):
        beta, retVal, flags = optimize.fmin_l_bfgs_b(
            func=self.maximumLikelihoodBeta,
            x0=self.beta,
            fprime=self.gradBeta,
            args=(),
            m=5,
            factr=BETA_FCTR,
            maxiter=BETA_ITERATIONS,
        )
        converged = True
        if flags["warnflag"] != 0:
            converged = False
        self.logger.info("Beta converged? : %s", 'yes' if converged else 'no')
        return beta.reshape((self.aspect_cnt, self.words_cnt)), converged


    def maximumLikelihoodAlphaHat(self, x, *args):
        alpha_d_hat = x.reshape((self.aspect_cnt, 1))
        # FIXME: alpha_d_hat is too big
        print('exp alpha', np.exp(alpha_d_hat))
        alpha_d = (np.exp(alpha_d_hat) / np.sum(np.exp(alpha_d_hat))).reshape(self.aspect_cnt, )
        rd, Sd, deltasq, mu, sigmaInv, pi = args

        term1 = np.einsum('i,i->', alpha_d.reshape(self.aspect_cnt, ), Sd.reshape(self.aspect_cnt, )) - rd
        term1 /= deltasq
        term1 = -1 * term1 * term1
        term2 = -1 * self.pi * np.einsum('i,i->', alpha_d, np.square(Sd.reshape(self.aspect_cnt, ) - rd))
        temp3 = alpha_d_hat - mu
        term3 = -1 * np.dot(np.dot(temp3.transpose(), sigmaInv), temp3)[0][0]
        print('maximumLikelihoodAlphaHat', term1 + term2 + term3)
        return term1 + term2 + term3

    def gradAlphaHat(self, x, *args):
        alpha_d_hat = x.reshape((self.aspect_cnt, 1))
        alpha_d = np.exp(alpha_d_hat) / np.sum(np.exp(alpha_d_hat))
        rd, Sd, deltasq, mu, sigmaInv, pi = args

        off_diag_terms = Sd * alpha_d
        off_diag_sum = np.sum(off_diag_terms)

        off_diag_sq_terms = np.square(Sd - rd) * alpha_d
        off_diag_sq_sum = np.sum(off_diag_sq_terms)

        sumterm1 = Sd * (1 - alpha_d) - off_diag_sum + off_diag_terms

        sumterm2 = np.square(Sd - rd) * (1 - alpha_d) - off_diag_sq_sum + off_diag_sq_terms
        dasda = alpha_d * sumterm1

        term1 = np.einsum('i,i->', alpha_d.reshape(self.aspect_cnt, ), Sd.reshape(self.aspect_cnt, )) - rd
        term1 = -((2 * term1 * dasda) / deltasq).reshape((self.aspect_cnt,))

        term2 = (-pi * alpha_d * sumterm2).reshape((self.aspect_cnt,))

        term3 = -2 * np.einsum(
            'ij,j->i',
            self.sigmaInv,
            alpha_d_hat.reshape(self.aspect_cnt, ) - mu.reshape(self.aspect_cnt, )
        )

        assert term1.shape == term2.shape
        assert term2.shape == term3.shape
        all_terms = term1 + term2 + term3
        print('~ grad alpha ~')
        print('alphah', alpha_d_hat.reshape(self.aspect_cnt, ))
        print('mu', mu.reshape(self.aspect_cnt, ))
        print('alphah - mu', alpha_d_hat.reshape(self.aspect_cnt, ) - mu.reshape(self.aspect_cnt, ))
        return all_terms

    def calc_alpha_d_hat(self, d):
        alpha_d_hat = self.alpha_hat[:, d].reshape((self.aspect_cnt, 1))
        alpha_d = self.alpha[:, d].reshape((self.aspect_cnt, 1))
        review_idx = self.train_indices[d]
        rd = float(self.aspect_ratings[review_idx]["Overall"])
        Sd = self.S[:, d].reshape((self.aspect_cnt, 1))
        Args = (rd, Sd, self.delta_sq, self.mu, self.sigmaInv, self.pi)

        print('alpha before opt >>>>', alpha_d_hat)
        alpha_d_hat, retVal, flags = optimize.fmin_l_bfgs_b(
            func=self.maximumLikelihoodAlphaHat,
            x0=alpha_d_hat,
            fprime=self.gradAlphaHat,
            args=Args,
#             m=5,
            factr=ALPHA_FCTR,
            maxiter=ALPHA_ITERATIONS,
        )
        converged = True
        if flags["warnflag"] != 0:
            converged = False

        # self.logger.info("Alpha converged? : %s", 'yes' if converged else 'no')
        # Normalizing alpha_d so that it follows dirichlet distribution
#         alpha_d = np.exp(alpha_d_hat) / (np.sum(np.exp(alpha_d_hat)))

        return alpha_d_hat.reshape((self.aspect_cnt,)), converged

    def beta_likelihood(self):
        return -1.0 * self.lambda_param * np.sum(np.einsum('ij,ij->i',self.beta,self.beta))


    def dataLikelihood(self):
        likelihood = 0.0
        for d in range(self.train_reviews_cnt):
            review_idx = self.train_indices[d]
            Rd = float(self.aspect_ratings[review_idx]["Overall"])
            Sd = self.S[:, d].reshape((self.aspect_cnt,))
            alpha_d = self.alpha[:, d].reshape((self.aspect_cnt,))
            temp = (self.calc_overall_rating(alpha_d, Sd) - Rd) / self.delta_sq
            try:
                likelihood += temp * temp
            except Exception:
                self.logger.debug("Exception in dataLikelihood")
        return -1 * likelihood

    def alpha_likelihood(self):
        likelihood = 0.0
        for d in range(self.train_reviews_cnt):
#             alpha_d = self.alpha[:, d].reshape((self.aspect_cnt, 1))
            alpha_d_hat = self.alpha_hat[:, d].reshape((self.aspect_cnt, 1))
            temp2 = alpha_d_hat - self.mu
            temp2 = np.dot(np.dot(temp2.transpose(), self.sigmaInv), temp2)[0]
            likelihood += temp2
        try:
            likelihood += np.log(np.linalg.det(self.sigma))
        except FloatingPointError:
            self.logger.debug(
                "Exception in alpha_likelihood: %f", np.linalg.det(self.sigma)
            )
        return -1.0 * likelihood

    def aux_likelihood(self):
        likelihood = 0.0
        for d in range(self.train_reviews_cnt):
            alpha_d = self.alpha[:, d].reshape((self.aspect_cnt, 1))
            review_idx = self.train_indices[d]
            rd = float(self.aspect_ratings[review_idx]["Overall"])
            Sd = self.S[:, d].reshape((self.aspect_cnt, 1))
            likelihood += np.einsum('i,i->', alpha_d.reshape(self.aspect_cnt, ), np.square(rd - Sd.reshape(self.aspect_cnt, )))
        likelihood = -1.0 * self.pi * likelihood
        return likelihood


    def calc_likelihood(self):
        return \
            -np.log(self.delta_sq) + \
            self.dataLikelihood() + \
            self.alpha_likelihood() + \
            self.beta_likelihood() + \
            self.aux_likelihood()

    # Expectation calculation step
    def EStep(self):
        for d in range(self.train_reviews_cnt):
            review_idx = self.train_indices[d]
            W = self.Wd[review_idx]
            self.S[:, d] = self.calc_aspect_ratings(W)
#             print(self.S[:, d])

            alpha_d_hat, converged = self.calc_alpha_d_hat(d)

            if converged:
                print('alpha updating >>>>', alpha_d_hat)
                self.alpha_hat[:, d] = alpha_d_hat
                self.alpha[:, d] = np.exp(alpha_d_hat) / np.sum(np.exp(alpha_d_hat))
            else:
                print(f"alpha not converged for review {review_idx}")

        if self.should_assert:
            assert_alpha(self.alpha)

    # Maximization step
    def MStep(self):
        self.mu = self.calc_mu()
        self.logger.info("Mu calculated")

        self.calc_sigma()
        self.logger.info("Sigma calculated : %s " % np.linalg.det(self.sigma))

        beta, converged = self.calcBeta()
        if converged:
            self.beta = beta
        self.logger.info("Beta calculated")

        self.delta_sq = self.calc_delta_square()
        self.logger.info("Delta_sq calculated")

    def EMAlgo(self, maxIter, covergence_threshold):
        self.logger.info("Training started")
        iteration = 0
        # FIXME: fill in self.S
        self.EStep()
        old_likelihood = self.calc_likelihood()

        diff = np.Inf
        while (iteration < max(8, maxIter) and abs(diff) > covergence_threshold):
            self.EStep()
            self.logger.info("EStep completed")

            self.MStep()

            likelihood = self.calc_likelihood()

            self.logger.info("MStep completed")

            diff = (likelihood - old_likelihood) # old_likelihood
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
            print("Predicted Rating:", (overall_rating))
            print("Actual vs Predicted Aspect Ratings:")
            for aspect, rating in self.aspect_ratings[review_idx].items():
                if (
                    aspect != "Overall"
                    and aspect.lower() in self.aspect_index_mapping.keys()
                ):
                    r = self.aspect_index_mapping[aspect.lower()]
                    print("  Aspect: %15s | Rating: %s | Predicted: %.1f" % (aspect, rating, Sd[r]))
            print("")

# eq (11) in paper, but flipped signs for minimization
def maximum_likelihood_beta(
    alpha, # [aspect X words]
    beta, # [aspect X words]
    delta_sq, # scalar
    pi, #scalar
    lambda_param,
    aspect_ratings, # [reviews]
    train_indices, # [reviews]
    Wd, # [reviews X aspect X words]
    S,
    words_cnt,
):
    aspect_cnt = len(alpha)
    train_reviews_cnt = len(train_indices)

    beta = beta.reshape((aspect_cnt, words_cnt))
    inner_bracket = np.empty(shape=train_reviews_cnt)

    term2 = 0
    term1 = 0

    for d in range(train_reviews_cnt):

        alpha_d = alpha[:, d].reshape((aspect_cnt, 1))
        review_idx = train_indices[d]
        rd = float(aspect_ratings[review_idx]["Overall"])
        Sd = S[:, d].reshape((aspect_cnt,))
        term2 += np.einsum('i,i->', alpha_d.reshape(aspect_cnt, ), np.square(Sd.reshape(aspect_cnt, ) - rd))
        temp = np.einsum('i,i->', alpha_d.reshape(aspect_cnt, ), Sd.reshape(aspect_cnt, )) - rd
        term1 += temp * temp

    term1 /= -1.0 * delta_sq
    term2 = -1.0 * pi * term2
    term3 = -1.0 * lambda_param * np.sum(np.einsum('ij,ij->i', beta, beta))

    return term1 + term2 + term3

def maximum_likelihood_beta_grad(
    alpha, # [aspect X words]
    beta, # [aspect X words]
    delta_sq, # scalar
    pi, #scalar
    lambda_param,
    aspect_ratings, # [reviews]
    train_indices, # [reviews]
    Wd, # [reviews X aspect X words]
    S,
    words_cnt,
): # [aspect X words]
    aspect_cnt = len(alpha)
    train_reviews_cnt = len(train_indices)
    beta = beta.reshape((aspect_cnt, words_cnt))

    grad_beta_mat = np.empty(shape=((aspect_cnt, words_cnt)), dtype="float64")
    for i in range(aspect_cnt):
        grad_beta_i = np.zeros(shape=(1, words_cnt))
        beta_i = beta[i, :]
        for d in range(train_reviews_cnt):
            review_idx = train_indices[d]  # review index in wList
            W = Wd[review_idx]
            rd = float(aspect_ratings[review_idx]["Overall"])
            alpha_d = alpha[:, d].reshape((aspect_cnt, 1))
            Sd = S[:, d].reshape((aspect_cnt,))

            inner_bracket = np.einsum('i,i->', alpha_d.reshape(aspect_cnt, ), Sd.reshape(aspect_cnt, )) - rd
            inner_bracket /= delta_sq
            inner_bracket += pi * (Sd[i] - rd)

            dsdb = Sd[i] * W[i, :]
            grad_beta_i += alpha_d[i] * inner_bracket * dsdb
        grad_beta_mat[i, :] = grad_beta_i
        term3 = lambda_param * beta_i
    return -2.0 * grad_beta_mat.reshape((aspect_cnt * words_cnt,))


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
