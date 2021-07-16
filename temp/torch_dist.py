# http://stillbreeze.github.io/REINFORCE-vs-Reparameterization-trick/
import torch
from torch import distributions

idx = 1
for dist_str in dir(distributions):
    if dist_str[0].isupper() and "Transform" not in dist_str and dist_str not in ["Distribution", "ExponentialFamily", "Independent", "MixtureSameFamily"]:
        if dist_str in ["HalfCauchy", "HalfNormal"]:
            instance = eval("torch.distributions." + dist_str)(scale=torch.tensor([0.5]))
        elif dist_str in ["FisherSnedecor"]:
            instance = eval("torch.distributions." + dist_str)(df1=torch.tensor([0.5]), df2=torch.tensor([0.5]))
        elif dist_str in ["Exponential", "Poisson"]:
            instance = eval("torch.distributions." + dist_str)(rate=torch.tensor([0.5]))
        elif dist_str in ["Dirichlet"]:
            instance = eval("torch.distributions." + dist_str)(concentration=torch.tensor([0.5]))
        elif dist_str in ["Gamma"]:
            instance = eval("torch.distributions." + dist_str)(concentration=torch.tensor([0.5]), rate=torch.tensor([0.5]))
        elif dist_str in ["Chi2", "StudentT"]:
            instance = eval("torch.distributions." + dist_str)(df=torch.tensor([0.5]))
        elif dist_str in ["Categorical", "Geometric"]:
            instance = eval("torch.distributions." + dist_str)(probs=torch.tensor([0.5]))
        elif dist_str in ["Binomial", "Multinomial", "NegativeBinomial"]:
            instance = eval("torch.distributions." + dist_str)(total_count=1, probs=torch.tensor([0.5]))
        elif dist_str in ["Beta", "Kumaraswamy"]:
            instance = eval("torch.distributions." + dist_str)(concentration1=torch.tensor([0.1]), concentration0=torch.tensor([0.1]))
        elif dist_str in ["Bernoulli", "ContinuousBernoulli", "OneHotCategorical", "OneHotCategoricalStraightThrough"]:
            instance = eval("torch.distributions." + dist_str)(probs=torch.tensor([0.5]))
        elif dist_str in ["LKJCholesky"]:
            instance = eval("torch.distributions." + dist_str)(dim=2, concentration=torch.tensor([0.5]))
        elif dist_str in ["LowRankMultivariateNormal"]:
            instance = eval("torch.distributions." + dist_str)(loc=torch.zeros(2), cov_factor=torch.tensor([[1.], [0.]]), cov_diag=torch.ones(2))
        elif dist_str in ["MultivariateNormal"]:
            instance = eval("torch.distributions." + dist_str)(loc=torch.zeros(2), covariance_matrix=torch.eye(2))
        elif dist_str in ["Pareto"]:
            instance = eval("torch.distributions." + dist_str)(scale=torch.tensor([1.0]), alpha=torch.tensor([1.0]))
        elif dist_str in ["RelaxedBernoulli", "RelaxedOneHotCategorical"]:
            instance = eval("torch.distributions." + dist_str)(temperature=torch.tensor([1.0]), probs=torch.tensor([1.0]))
        elif dist_str in ["Uniform"]:
            instance = eval("torch.distributions." + dist_str)(low=torch.tensor([0.0]), high=torch.tensor([1.0]))
        elif dist_str in ["VonMises"]:
            instance = eval("torch.distributions." + dist_str)(loc=torch.tensor([0.0]), concentration=torch.tensor([1.0]))
        elif dist_str in ["Weibull"]:
            instance = eval("torch.distributions." + dist_str)(scale=torch.tensor([1.0]), concentration=torch.tensor([1.0]))
        else:
            instance = eval("torch.distributions." + dist_str)(loc=torch.tensor([0.5]), scale=torch.tensor([0.5]))

        print(idx, hasattr(instance, "log_prob"), hasattr(instance, "entropy"), hasattr(instance, "rsample"), instance)
        idx += 1

###
# 1 True True True Bernoulli(probs: tensor([0.5000]))
# 2 True True True Beta()
# 3 True True True Binomial(total_count: tensor([1.]), probs: tensor([0.5000]))
# 4 True True True Categorical(probs: tensor([1.]))
# 5 True True True Cauchy(loc: tensor([0.5000]), scale: tensor([0.5000]))
# 6 True True True Chi2()
# 7 True True True ContinuousBernoulli(probs: tensor([0.5000]))
# 8 True True True Dirichlet(concentration: tensor([0.5000]))
# 9 True True True Exponential(rate: tensor([0.5000]))
# 10 True True True FisherSnedecor(df1: tensor([0.5000]), df2: tensor([0.5000]))
# 11 True True True Gamma(concentration: tensor([0.5000]), rate: tensor([0.5000]))
# 12 True True True Geometric(probs: tensor([0.5000]))
# 13 True True True Gumbel(loc: tensor([0.5000]), scale: tensor([0.5000]))
# 14 True True True HalfCauchy()
# 15 True True True HalfNormal()
# 16 True True True Kumaraswamy(concentration1: tensor([0.1000]), concentration0: tensor([0.1000]))
# 17 True True True LKJCholesky(concentration: tensor([0.5000]))
# 18 True True True Laplace(loc: tensor([0.5000]), scale: tensor([0.5000]))
# 19 True True True LogNormal()
# 20 True True True LogisticNormal()
# 21 True True True LowRankMultivariateNormal(loc: torch.Size([2]), cov_factor: torch.Size([2, 1]), cov_diag: torch.Size([2]))
# 22 True True True Multinomial()
# 23 True True True MultivariateNormal(loc: torch.Size([2]), covariance_matrix: torch.Size([2, 2]))
# 24 True True True NegativeBinomial(total_count: tensor([1.]), probs: tensor([0.5000]))
# 25 True True True Normal(loc: tensor([0.5000]), scale: tensor([0.5000]))
# 26 True True True OneHotCategorical()
# 27 True True True OneHotCategoricalStraightThrough()
# 28 True True True Pareto(alpha: tensor([1.]), scale: tensor([1.]))
# 29 True True True Poisson(rate: tensor([0.5000]))
# 30 True True True RelaxedBernoulli()
# 31 True True True RelaxedOneHotCategorical()
# 32 True True True StudentT(df: tensor([0.5000]), loc: tensor([0.]), scale: tensor([1.]))
# 33 True True True Uniform(low: tensor([0.]), high: tensor([1.]))
# 34 True True True VonMises(loc: tensor([0.]), concentration: tensor([1.]))
# 35 True True True Weibull(scale: tensor([1.]), concentration: tensor([1.]))
###