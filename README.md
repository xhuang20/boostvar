
<!-- README.md is generated from README.Rmd. Please edit that file -->

# boostvar: An R package for computing the p-values in high-dimensional vector autoregression

<!-- badges: start -->
<!-- badges: end -->

This R package computes the least-squares boosting estimates, standard
errors, and p-values for vector autoregression models.

This package also offers a function **boostls.R** to compute the
estimates, standard errors, and p-values for a cross-section regression.

## Installation

You can install **boostvar** from
[GitHub](https://github.com/xhuang20/boostvar.git) with:

``` r
devtools::install_github("xhuang20/boostvar")
```

## An example of vector autoregression

We give an example for a bivariate VAR model.

``` r
library(boostvar)
# An example of bivariate vector autoregression
set.seed(123)
p = 2
d = 2
nobs = 100
bstop = 50
y = matrix(0, nobs, d)
b0 = matrix(c(0.02,0.03),2,1)
b1 = matrix(c(0.5,0.4,0.1,0.5),2,2)
b2 = matrix(c(0,0.25,0,0),2,2)
for (i in (p+1):dim(y)[1]) {
      y[i,] = t(b0) + t(b1 %*%  y[i-1,]) + t(b2 %*% y[i-2,]) +
      c(rnorm(1,0,0.3), rnorm(1,0,0.2))
}
y = tail(y,-p)
result = boostvar(y, p = p, bstop = bstop, nu = 0.1, group.type = 1)
dim(result$beta)
#> [1]  2  2  2 50
dim(result$pval)
#> [1]  2  2  2 50
```

The object *result* includes all LS-Boost estimates at each of the 50
steps and their p-values. The component *result$beta* is a 4-dimensional
array, where the 3rd dimension corresponds to each of the lag in a VAR
and the 4th dimension corresponds to each boosting step. Dimensions 1
and 2 are the same as the number of variables in a VAR so that, for
example, *result$beta\[,,2,3\]* is the estimated square coefficient
matrix for the 2nd lag at boosting step 3. Values in the component
*result$se* are the standard errors that can be used to construct
confidence intervals.

## An example of cross-section regression

We use the red wine quality data from
\[<https://archive.ics.uci.edu/ml/datasets/wine+Quality>\] as an example
in cross-section regression.

``` r
data.link = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
redwine = read.csv(data.link, header = TRUE, sep = ';')
redwine = redwine[1:100,]   # Try the first 100 observations.
result = boostls(y = redwine$quality,
                 x = subset(redwine, select = -c(quality)),
                 bstop = 50,
                 nu = 0.1)
dim(result$beta)
#> [1] 11 50
dim(result$pval)
#> [1] 11 50
```

The component *result$beta* stores the boosting estimates for 11
regressors in 50 steps. The component *result$pval* gives p-values for
all nonzero estimates in *result$beta*. Again, the standard errors in
*result$se* can be used to compute the confidence intervals.

## References

Huang, X. (2022) Boosted p-values for high-dimensional vector
autoregression, working paper.
