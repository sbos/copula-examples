import numpy as np
from scipy.stats import norm, multivariate_normal, mvn


def log_gaussian_copula(u, cor):
    a = np.linalg.inv(cor) - np.eye(cor.shape[0])
    x = norm.ppf(u)
    x *= np.dot(x, a)
    s = np.sum(x, axis=1)
    return -0.5 * (s + np.linalg.slogdet(cor)[1])


def log_gaussian_pdf(x, mu, sigma):
    diag_sigma = np.sqrt(np.diag(sigma))
    r = np.sum(norm.logpdf(x, loc=mu, scale=diag_sigma), axis=1)
    cor = sigma / diag_sigma[:, np.newaxis] / diag_sigma[np.newaxis, :]
    u = norm.cdf(x, loc=mu, scale=diag_sigma)
    u[u >= 1.] -= 1e-10
    cop = log_gaussian_copula(u, cor)
    return r + cop


def decorrelated_samples(x1, x2, mu, sigma, rho):
    n1 = norm(loc=mu[0], scale=np.sqrt(sigma[0, 0]))
    n2 = norm(loc=mu[1], scale=np.sqrt(sigma[1, 1]))
    u1 = n1.cdf(x1[:, 0]) * n2.cdf(x1[:, 1])
    u2 = n1.cdf(x2[:, 0]) * n2.cdf(x2[:, 1])

    cor = np.ones((2, 2))
    cor[0, 1] = cor[1, 0] = rho
    return np.exp(log_gaussian_copula(np.vstack([u1, u2]).T, cor) +
                  n1.logpdf(x2[:, 0]) + n2.logpdf(x2[:, 1]))


def copula_density(x1, x2, mu, sigma, rho):
    n1 = norm(loc=mu[0], scale=np.sqrt(sigma[0, 0]))
    n2 = norm(loc=mu[1], scale=np.sqrt(sigma[1, 1]))
    u1 = n1.cdf(x1[:, 0]) * n2.cdf(x1[:, 1])
    u2 = n1.cdf(x2[:, 0]) * n2.cdf(x2[:, 1])

    cor = np.ones((2, 2))
    cor[0, 1] = cor[1, 0] = rho
    return np.exp(log_gaussian_copula(np.vstack([u1, u2]).T, cor))


def joint_samples(x1, x2, mu, sigma, rho):
    n1 = norm(loc=mu[0], scale=np.sqrt(sigma[0, 0]))
    n2 = norm(loc=mu[1], scale=np.sqrt(sigma[1, 1]))
    u1 = n1.cdf(x1[:, 0]) * n2.cdf(x1[:, 1])
    u2 = n1.cdf(x2[:, 0]) * n2.cdf(x2[:, 1])

    cor = np.ones((2, 2))
    cor[0, 1] = cor[1, 0] = rho
    return log_gaussian_copula(np.vstack([u1, u2]).T, cor) + \
                  n1.logpdf(x2[:, 0]) + n2.logpdf(x2[:, 1]) + \
                  n1.logpdf(x1[:, 0]) + n2.logpdf(x1[:, 1])

mu = np.zeros(2)
rho = 0.
sigma = np.array([[2., rho], [rho, 3.]])

import matplotlib.pyplot as plt

# xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
# coord_grid = np.c_[xx.ravel(), yy.ravel()]
# f, (ax1, ax2) = plt.subplots(1, 2)
# dens1 = np.exp(log_gaussian_pdf(coord_grid, mu, sigma).reshape(xx.shape))
# dens2 = multivariate_normal.pdf(coord_grid, mean=mu, cov=sigma).reshape(xx.shape)
#
# print np.sum(np.square(dens1 - dens2))
# ax1.contourf(xx, yy, dens1)
# ax2.contourf(xx, yy, dens2)
# plt.show()

xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
coord_grid = np.c_[xx.ravel(), yy.ravel()]

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.contourf(xx, yy, multivariate_normal.pdf(coord_grid, mean=mu, cov=sigma).reshape(xx.shape), 30)

sample1 = np.array([[-1., -2.]])
sample1 = np.tile(sample1, [coord_grid.shape[0], 1])
sample2 = coord_grid

ax2.contourf(xx, yy, decorrelated_samples(sample1, sample2, mu, sigma, -0.9).reshape(xx.shape), 30)
ax2.scatter(sample1[0, 0], sample1[0, 1])
ax1.scatter(sample1[0, 0], sample1[0, 1])
ax2.set_xlim(xx.min(), xx.max())
ax2.set_ylim(yy.min(), yy.max())
ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())
f.set_size_inches(20, 10, True)

#plt.show()

sample1 = multivariate_normal.rvs(mu, sigma, 500000)
sample2 = multivariate_normal.rvs(mu, sigma, 500000)

original_weights = copula_density(sample1, sample2, mu, sigma, -0.9)

data = np.vstack([sample1, sample2])

weights = np.tile(original_weights, [2])

print data.shape, weights.shape

f = plt.figure()
f.set_size_inches(5, 5)
ax = f.add_subplot(1, 1, 1)
ax.hist2d(data[:, 0], data[:, 1], weights=weights, bins=100)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
#plt.show()

data = np.hstack([sample1, sample2])
sample_mean = np.sum(data * original_weights[:, None], axis=0) / original_weights.sum()
sample_cov = np.cov(data.T, aweights=original_weights)
mvn4d = multivariate_normal(sample_mean,
                            sample_cov)

print sample_mean
print sample_cov

print np.dot(mvn4d.logpdf(data), original_weights) / original_weights.sum()
print np.dot(joint_samples(sample1, sample2, mu, sigma, -0.9), original_weights) / original_weights.sum()
