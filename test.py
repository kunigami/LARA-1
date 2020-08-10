import unittest
import numpy as np
import random

from LRR import maximum_likelihood_beta, maximum_likelihood_beta_grad


class LLRTest(unittest.TestCase):

    def testBetaFunctionHardCodedData(self):
        words_cnt = 5

        Wd = [
            # review 1
            np.array([
                [0.1, 0.2, 0.3, 0.4, 0.0],
                [0.0, 0.1, 0.0, 0.0, 0.9],
            ]),
            # review 2
            np.array([
                [0.1, 0.2, 0.3, 0.4, 0.0],
                [0.0, 0.1, 0.0, 0.0, 0.9],
            ]),
            # review 3
            np.array([
                [0.1, 0.2, 0.3, 0.4, 0.0],
                [0.0, 0.1, 0.0, 0.0, 0.9],
            ]),
        ]
        indices = list(range(len(Wd)))

        alpha = np.array([
            [0.1, 0.3, 0.5],
            [0.9, 0.7, 0.5],
        ])

        beta = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.0],
            [0.0, 0.1, 0.0, 0.0, 0.9],
        ])

        aspect_ratings = [
            {
                "Overall": 1,
            },
            {
                "Overall": 5,
            },
            {
                "Overall": 3,
            },
        ]

        score = maximum_likelihood_beta(
            alpha=alpha,
            beta=beta,
            delta=1.0,
            aspect_ratings=aspect_ratings,
            train_indices=indices,
            Wd=Wd,
            words_cnt=words_cnt
        )

        self.assertAlmostEqual(score, 12.404, delta=0.01)

        beta_grad = maximum_likelihood_beta_grad(
            alpha=alpha,
            beta=beta,
            aspect_ratings=aspect_ratings,
            train_indices=indices,
            Wd=Wd,
            words_cnt=words_cnt
        )

        beta_grad_expected = [
            -0.2544,
            -0.5088,
            -0.7632,
            -1.0176,
            0.,
            0.,
            -0.4464,
            0.,
            0.,
            -4.0176
        ]

        for i in range(len(beta_grad)):
            self.assertAlmostEqual(beta_grad[i], beta_grad_expected[i], delta=0.01)


    def testBetaFunctionRandomData(self):
        random.seed(1)
        np.random.seed(1)

        score = self._runBetaWithRandomData(
            review_cnt=10,
            aspect_cnt=7,
            words_cnt=30
        )

        self.assertAlmostEqual(score, 53.43, delta=0.01)

    # Performance test
    def testBetaFunctionWithLargeRandomData(self):
        for i in range(30):
            score = self._runBetaWithRandomData(
                review_cnt=1000,
                aspect_cnt=10,
                words_cnt=100
            )

        self.assertTrue(True)

    # Performance test
    def testBetaGradFunctionWithLargeRandomData(self):
        for i in range(30):
            score = self._runBetaGradWithRandomData(
                review_cnt=1000,
                aspect_cnt=10,
                words_cnt=100
            )

        self.assertTrue(True)

    def getRandomData(self, review_cnt, aspect_cnt, words_cnt):
        alpha = np.random.dirichlet(np.ones(aspect_cnt), size=1).reshape(aspect_cnt, 1)
        for i in range(review_cnt - 1):
            alpha = np.hstack(
                (
                    alpha,
                    np.random.dirichlet(np.ones(aspect_cnt), size=1).reshape(aspect_cnt, 1),
                )
            )

        beta = np.random.uniform(low=-0.1, high=0.1, size=(aspect_cnt, words_cnt))

        aspect_ratings = [{"Overall": random.randint(1, 5)} for d in range(review_cnt)]

        indices = list(range(review_cnt))

        Wd = []
        for d in range(review_cnt):
            freq_by_aspect = []
            for k in range(aspect_cnt):
                freq = np.random.rand(words_cnt)
                norm = np.linalg.norm(freq)
                freq = freq / norm
                freq_by_aspect.append(freq)

            W = np.array(freq_by_aspect)
            Wd.append(W)

        return [
            alpha,
            beta,
            aspect_ratings,
            indices,
            Wd,
        ]


    def _runBetaWithRandomData(self, review_cnt, aspect_cnt, words_cnt):
        [alpha, beta, aspect_ratings, train_indices, Wd] = self.getRandomData(
            review_cnt, aspect_cnt, words_cnt
        )

        score = maximum_likelihood_beta(
            alpha=alpha,
            beta=beta,
            delta=1.0,
            aspect_ratings=aspect_ratings,
            train_indices=train_indices,
            Wd=Wd,
            words_cnt=words_cnt
        )

        return score


    def _runBetaGradWithRandomData(self, review_cnt, aspect_cnt, words_cnt):
        [alpha, beta, aspect_ratings, train_indices, Wd] = self.getRandomData(
            review_cnt, aspect_cnt, words_cnt
        )

        grad = maximum_likelihood_beta_grad(
            alpha=alpha,
            beta=beta,
            aspect_ratings=aspect_ratings,
            train_indices=train_indices,
            Wd=Wd,
            words_cnt=words_cnt
        )

        return grad

if __name__ == '__main__':
    unittest.main()
