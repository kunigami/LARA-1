import numpy as np
from scipy import optimize
import json
import random
import logging

from typing import Any, List, Dict, Optional, Tuple

modelDataDir = "modelData/"

# BETA_ITERATIONS = 150000
# For floating point comparison
EPS = 0.0001

BETA_ITERATIONS = 500
ALPHA_ITERATIONS = 10
# For floating point comparison
LAMBDA = 2 # regularization parameter for beta
BETA_FCTR = 1e12
ALPHA_FCTR = 1e12
PI = 0.5
# Typical values for factr are:
# 1e12 for low accuracy; 1e7 for moderate accuracy; 10.0 for extremely high accuracy.

t_aspect = str
t_word = str

class ModelParams():

    # mean parameter for the Gaussian distribution for the aspect weights alpha (aspects X 1)
    mu: List[float]
    # sigma - variance parameter for the Gaussian distribution (aspects x aspects)
    sigma: Any
    # cache of sigma^{-1}
    sigma_inv: Any
    # delta^2 should be the variance of normal dist rd is draw from
    # represent uncertainty of overall rating predictions
    delta_sq: float
    # matrix Beta for the whole corpus (for all aspects, for all words)  [aspects X words]
    # beta is word sentiment polarity on that aspect
    beta: Any

    def __init__(self, aspect_cnt: int, words_cnt: int):
        self.delta_sq = 1.0

        # self.mu = np.random.dirichlet(np.ones(aspect_cnt), size=1).reshape(aspect_cnt, 1)
        self.mu = np.random.uniform(low=-1., high=1., size=(aspect_cnt, 1))

        self.beta = np.random.uniform(low=-0.1, high=0.1, size=(aspect_cnt, words_cnt))

        self.set_sigma(np.eye(aspect_cnt))

    def set_sigma(self, sigma):
        self.sigma = sigma
        self.sigma_inv = np.linalg.inv(self.sigma)

class LRR():

    lambda_param: float
    should_assert: bool
    word_index_mapping:  Dict[str, int]
    aspect_index_mapping: Dict[t_aspect, int]
    word_correlation_by_aspect: List[Dict[t_aspect, Dict[t_word, int]]]
    aspect_ratings: List[Dict[t_aspect, float]]
    reviews_ids: List[str]
    params: ModelParams
    # reviews_cnt X aspect_cnt matrix
    alpha: List[float]

    def __init__(self, should_assert: bool=False):
        random.seed(0)

        self.lambda_param = LAMBDA
        self.should_assert = should_assert

        self.initialize_input_from_file()

        [self.train_indices, self.test_indices] = self.split_reviews_into_training_and_test(self.reviews_cnt)

        self.params = ModelParams(self.aspect_cnt, self.words_cnt())

        # matrix of aspect rating vectors (Sd) of all reviews - [aspects X reviews]
        # Aspect weight vector
        self.S = np.empty(shape=(self.aspect_cnt, self.training_size()), dtype=np.float64)

        self.Wd = []
        for d in range(self.reviews_cnt):
            self.Wd.append(self.createWMatrix(self.word_correlation_by_aspect[d]))
        if should_assert:
            assert_words_matrix(self.Wd, reviews_cnt=self.reviews_cnt, aspect_cnt=self.aspect_cnt)

        self.alpha = np.empty(shape=(self.training_size(), self.aspect_cnt))
        for d in range(self.training_size()):
            self.alpha[d] = np.ones(self.aspect_cnt)/self.aspect_cnt

        if self.should_assert:
            assert_alpha(self.alpha)

        """ testing for positive semi definite
        if(np.all(np.linalg.eigvals(self.sigma) > 0)): #whether is positive semi definite
            print("yes")
        print(self.sigma)
        """
        self.setup_logger()

    def initialize_input_from_file(self):
        words = self.load_words()
        self.word_index_mapping = self.create_word_index_mapping(words)

        # Maps aspects to related keywords
        aspect_keywords = self.load_aspect_keywords()
        self.aspect_index_mapping = self.create_aspect_index_mapping(aspect_keywords)
        self.aspect_cnt = len(self.aspect_index_mapping)

        # Histogram of words for each review and aspect
        self.word_correlation_by_aspect = self.load_word_correlation_by_aspect()

        # List of ratings for each aspect belonging to a review, s_d ([reviews X aspects])
        self.aspect_ratings = self.load_aspect_ratings()

        # List of review IDs
        self.reviews_ids = self.load_reviews_ids()
        self.reviews_cnt = len(self.reviews_ids)

    # breaking dataset into 3:1 ratio, 3 parts for training and 1 for testing
    def split_reviews_into_training_and_test(self, reviews_cnt: int) -> Tuple[List[int], List[int]]:
        train_indices = random.sample(range(0, reviews_cnt), int(0.75 * reviews_cnt))
        test_indices = list(set(range(0, reviews_cnt)) - set(train_indices))
        return (train_indices, test_indices)

    def training_size(self) -> int:
        return len(self.train_indices)

    def setup_logger(self):
        self.logger = logging.getLogger("LRR")
        self.logger.setLevel(logging.INFO)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("output/lrr.log", mode='w')
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def loadDataFromFile(self, fileName):
        json_data = None
        with open(modelDataDir + fileName, "r") as fp:
            json_data = json.load(fp)
        return json_data

    # given a dictionary as in every index of self.word_correlation_by_aspect,
    #
    # creates a W matrix as was in the paper
    def createWMatrix(self, histo_by_aspect):
        W = np.zeros(shape=(self.aspect_cnt, self.words_cnt()))
        for aspect, cnt_by_words in histo_by_aspect.items():
            total_count = sum(cnt_by_words.values())
            for word, cnt in cnt_by_words.items():
                aspect_index = self.aspect_index_mapping[aspect]
                word_index = self.word_index_mapping[word]
                W[aspect_index][word_index] = 1.0 * cnt / total_count
        return W

    # Computing aspectRating array for a review given Wd->W matrix for review 'd'
    def calc_aspect_ratings_for_review(self, Wd):
        # Inner product over words
        Sd = np.einsum("ij,ij->i", self.params.beta, Wd).reshape((self.aspect_cnt,))
        Sd = np.exp(Sd)
        print(Sd)
        return Sd

    # calculates mu for (t+1)th iteration. Eq. 8 in the paper.
    def calc_mu(self):
        # FIXME
        return np.sum(self.alpha_hat, axis=1).reshape((self.aspect_cnt, 1)) / self.training_size()
        # return np.sum(self.alpha, axis=1).reshape((self.aspect_cnt, 1)) / self.training_size()

    # calculates sigma for (t+1)th iteration. Eq. 9 in the paper.
    def calc_sigma(self):
        self.sigma.fill(0)
        for d in range(self.training_size()):
            alpha_aspects = self.alpha[:, d].reshape((self.aspect_cnt, 1))
            alpha_aspects = alpha_aspects - self.mu
            self.sigma = self.sigma + np.dot(alpha_aspects, alpha_aspects.transpose())

        for k in range(self.aspect_cnt):
            self.sigma[k][k] = (1.0 + self.sigma[k][k]) / (1.0 + self.training_size())

        self.sigmaInv = np.linalg.inv(self.sigma)

    def calc_overall_rating(self, alpha_d, Sd):
        return np.einsum('i,i->', alpha_d, Sd)

    # calculates delta square for (t+1)th iteration. Eq. 10 in the paper.
    def calc_delta_square(self):
        delta = 0.0
        for d in range(self.training_size()):
            rd = float(self.aspect_ratings[self.train_indices[d]]["Overall"])

            alpha_d = self.alpha[:, d].reshape((self.aspect_cnt,))
            Sd = self.S[:, d].reshape((self.aspect_cnt,))
            delta += (rd - self.calc_overall_rating(alpha_d, Sd))**2
        return delta / self.training_size()

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
            words_cnt=self.words_cnt()
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
            words_cnt=self.words_cnt()
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
        return beta.reshape((self.aspect_cnt, self.words_cnt())), converged


    def alpha_likelihood(self, curr_solution, *args):
        rd, sd, deltasq, mu, sigma_inv = args

        alpha_d = curr_solution.reshape((self.aspect_cnt, 1))

        # alpha_d^T s_d
        term1 = np.einsum('i,i->', alpha_d.reshape(self.aspect_cnt, ), sd.reshape(self.aspect_cnt, ))
        # print('sd', sd)
        # print('alpha_d', alpha_d)
        # print('term1', term1)
        term1 = - (rd - term1)**2 / (2*deltasq)

        term2 = alpha_d - mu
        term2 = -1 * np.dot(np.dot(term2.transpose(), sigma_inv), term2)[0][0]
        # print('alpha_likelihood', term1 + term2)
        return term1 + term2

    def alpha_likelihood_gradient(self, curr_solution, *args):
        rd, sd, deltasq, mu, sigma_inv = args
        alpha_d = curr_solution.reshape((self.aspect_cnt, 1))

        # alpha_d^T s_d
        term1 = np.einsum('i,i->', alpha_d.reshape(self.aspect_cnt, ), sd.reshape(self.aspect_cnt, ))
        term1 = - (rd - term1)*sd / (deltasq)

        term2 = - np.dot(sigma_inv, alpha_d - mu)

        gradient = term1 + term2
        print('grad', gradient)
        return gradient.reshape(self.aspect_cnt, )

    def calc_alpha_for_review(self, d: int):
        alpha_d = self.alpha[d].reshape((self.aspect_cnt, 1))
        review_idx = self.train_indices[d]
        print(review_idx, len(self.aspect_ratings))
        rd = float(self.aspect_ratings[review_idx]["overall"])
        Sd = self.S[:, d].reshape((self.aspect_cnt, 1))
        Args = (
            rd,
            Sd,
            self.params.delta_sq,
            self.params.mu,
            self.params.sigma_inv
        )

        print('alpha_d', alpha_d)
        alpha_d, retVal, flags = optimize.fmin_l_bfgs_b(
            func=self.alpha_likelihood,
            x0=alpha_d,
            fprime=self.alpha_likelihood_gradient,
            args=Args,
            factr=ALPHA_FCTR,
            maxiter=ALPHA_ITERATIONS,
        )
        converged = True
        if flags["warnflag"] != 0:
            converged = False

        return alpha_d.reshape((self.aspect_cnt,)), converged

    def beta_likelihood(self):
        return -1.0 * self.lambda_param * np.sum(np.einsum('ij,ij->i',self.beta,self.beta))


    def dataLikelihood(self):
        likelihood = 0.0
        for d in range(self.training_size()):
            review_idx = self.train_indices[d]
            Rd = float(self.aspect_ratings[review_idx]["Overall"])
            Sd = self.S[:, d].reshape((self.aspect_cnt,))
            alpha_d = self.alpha[d].reshape((self.aspect_cnt,))
            temp = (self.calc_overall_rating(alpha_d, Sd) - Rd) / self.delta_sq
            try:
                likelihood += temp * temp
            except Exception:
                self.logger.debug("Exception in dataLikelihood")
        return -1 * likelihood

    def aux_likelihood(self):
        likelihood = 0.0
        for d in range(self.training_size()):
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

    def update_alpha(self):
        for d in range(self.training_size()):
            review_idx = self.train_indices[d]
            W = self.Wd[review_idx]
            self.S[:, d] = self.calc_aspect_ratings_for_review(W)

            alpha_d, converged = self.calc_alpha_for_review(d)

            if converged:
                print('alpha updating >>>>', alpha_d)
                self.alpha[d] = alpha_d
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

    def solve(self, maxIter, covergence_threshold):
        self.logger.info("Training started")

        iteration = 0
        self.update_alpha()
        old_likelihood = self.calc_likelihood()
        return

        diff = np.Inf
        while (iteration < max(8, maxIter) and abs(diff) > covergence_threshold):
            self.update_alpha()
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

    def load_words(self) -> List[str]:
        return self.loadDataFromFile("vocab.json")

    def create_word_index_mapping(self, words: List[str]) -> Dict[str, int]:
        word_index_mapping = {}
        for i in range(len(words)):
            word_index_mapping[words[i]] = i
        return word_index_mapping

    def words_cnt(self):
        return len(self.word_index_mapping)

    def load_aspect_keywords(self) -> Dict[str, List[str]]:
        return self.loadDataFromFile("aspectKeywords.json")

    def create_aspect_index_mapping(self, aspect_keywords: Dict[str, List[str]]) -> Dict[str, int]:
        aspect_index_mapping = {}
        aspects = list(aspect_keywords.keys())
        for i in range(len(aspects)):
            aspect_index_mapping[aspects[i]] = i
        return aspect_index_mapping

    def aspect_cnt() -> int:
        return len(self.aspect_index_mapping)

    def load_word_correlation_by_aspect(self) -> List[Dict[str, Dict[str, int]]]:
        return self.loadDataFromFile("wList.json")

    def load_aspect_ratings(self) -> List[Dict[t_aspect, float]]:
        aspect_ratings = self.loadDataFromFile("ratingsList.json")
        aspect_ratings_norm = []
        for aspect_rating in aspect_ratings:
            aspect_rating_norm = self.normalize_aspect_rating(aspect_rating)
            if aspect_rating_norm is None:
                raise Exception('Invalid aspect rating', aspect_rating)
            aspect_ratings_norm.append(aspect_rating_norm)
        return aspect_ratings_norm

    def normalize_aspect_rating(self, aspect_rating) -> Optional[Dict[t_aspect, float]]:
        aspect_rating_norm = {}
        for aspect, rating in aspect_rating.items():
            rating_norm = float(rating)
            if rating_norm < 0:
                return None
            aspect_rating_norm[aspect.lower()] = rating_norm
        return aspect_rating_norm

    def load_reviews_ids(self) -> List[str]:
        reviews_ids = self.loadDataFromFile("reviewIdList.json")
        assert len(reviews_ids) > 0, "Reviews should exist in reviewIdList.json"
        return reviews_ids

    def testing(self):
        mu = self.mu.reshape((self.aspect_cnt,))
        for i in range(10):
            # self.reviews_cnt - self.training_size()):
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
            assert geq(a, 0.0) and leq(a, 1.0), "0 <= alpha[d][k] <= 1"
            total += a
        assert is_almost(total, 1.0), "alpha[d] must add up to 1"


def is_almost(a, b):
    return abs(a - b) < EPS

def geq(a, b):
    return a > b - EPS

def leq(a, b):
    return a < b + EPS
