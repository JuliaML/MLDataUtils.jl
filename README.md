# MLDataUtils

[![Build Status](https://travis-ci.org/JuliaML/MLDataUtils.jl.svg?branch=master)](https://travis-ci.org/JuliaML/MLDataUtils.jl)

Utility package for generating, loading, and processing Machine Learning datasets

## Installation

```Julia
Pkg.add("MLDataUtils")
using MLDataUtils
```

## Feature Normalization

This package contains a simple model called `FeatureNormalizer`, that can
be used to normalize training and test data with the parameters computed
from the training data

```Julia
x = collect(-5:.1:5)
X = [x x.^2 x.^3]'

# Derives the model from the given data
cs = fit(FeatureNormalizer, X)

# Normalizes the given data using the derived parameters
X_norm = predict(cs, X)
```

```
3x101 Array{Float64,2}:
 -1.70647  -1.67235  -1.63822  -1.60409   …  1.56996  1.60409  1.63822  1.67235  1.70647
  2.15985   2.03026   1.90328   1.77893      1.65719  1.77893  1.90328  2.03026  2.15985
 -2.55607  -2.40576  -2.26145  -2.12303      1.99038  2.12303  2.26145  2.40576  2.55607
```

The underlying functions can also be used directly

### Center data

`μ = center!(X[, μ])`

Centers each row of `X` around the corresponding entry in the vector `μ`.
If `μ` is not specified then it defaults to `mean(X, 2)`.

### Rescale data

`μ, σ = rescale!(X[, μ])`

Centers each row of `X` around the corresponding entry in the vector `μ`
and then rescaled to be between -1 and 1.
If `μ` is not specified then it defaults to `mean(X, 2)`.

### Normalize data

`μ, σ = normalize!(X[, μ, σ])`

Centers each row of `X` around the corresponding entry in the vector `μ`
and then rescaled using the corresponding entry in the vector `σ`.
If `μ` is not specified then it defaults to `mean(X, 2)`.
If `σ` is not specified then it defaults to `std(X, 2)`.


## Basis Expansion

`X = DataUtils.expand_poly(x; degree = 5)`

Performs a polynomial basis expansion of the given `degree`
for the vector `x`.
The return value `X` is a matrix of size `(degree, length(x))`.

_Note_: all the features of `X` are centered and rescaled.


## Data Generators

### Noisy function

`x, y = noisy_function(fun, x; noise = 0.01, f_rand = randn)`

Generates a noisy response `y` for the given function `fun`
by adding `noise .* f_randn(length(x))` to the result of `fun(x)`.

### Noisy sin

`x, y = noisy_sin(n, start, stop; noise = 0.3, f_rand = randn)`

Generates `n` noisy equally spaces samples of a sinus from `start` to `stop`
by adding `noise .* f_randn(length(x))` to the result of `fun(x)`.

### Noisy polynomial

`x, y = noisy_poly(coef, x; noise = 0.01, f_rand = randn)`

Generates a noisy response for a polynomial of degree `length(coef)`
using the vector `x` as input and adding `noise .* f_randn(length(x))` to the result.
The vector `coef` contains the coefficients for the terms of the polynome.
The first element of `coef` denotes the coefficient for the term with
the highest degree, while the last element of `coef` denotes the intercept.

## Datasets

The package contains a few static datasets to serve as toy examples.

### The Iris Dataset

`X, y, vars = load_iris(n)`

Loads the first `n` (of 150) observations from the
Iris flower data set introduced by Ronald Fisher (1936).
The 4 by `n` matrix `X` contains the numeric measurements,
in which each individual column denotes an observation.
The vector `y` contains the class labels as strings.
The optional vector `vars` contains the names of the features (i.e. rows of `X`)

Check out [the wikipedia entry](https://en.wikipedia.org/wiki/Iris_flower_data_set)
for more information about the dataset.

### Example: noisy line

`x, y, vars = load_line()`

Loads an artificial example dataset for a noisy line.
It is particularly useful to explain under- and overfitting.
The vector `x` contains 11 equally spaced points between 0 and 1
The vector `y` contains `x ./ 2 + 1` plus some gaussian noise
The optional vector `vars` contains descriptive names for `x` and `y`

![noisy_line](https://cloud.githubusercontent.com/assets/10854026/13020766/75b321d4-d1d7-11e5-940d-25974efa0710.png)

### Example: noisy sin

`x, y, vars = load_sin()`

Loads an artificial example dataset for a noisy sin.
It is particularly useful to explain under- and overfitting.
The vector `x` contains equally spaced points between 0 and 2π
The vector `y` contains `sin(x)` plus some gaussian noise
The optional vector `vars` contains descriptive names for `x` and `y`

![noisy_sin](https://cloud.githubusercontent.com/assets/10854026/13020842/eb6f2f30-d1d7-11e5-8a2c-a264fc14c861.png)

### Example: noisy polynomial

`x, y, vars = load_poly()`

Loads an artificial example dataset for a noisy quadratic function.
It is particularly useful to explain under- and overfitting.
The vector `x` contains 50 points between 0 and 4
The vector `y` contains `2.6 * x^2 + .8 * x` plus some gaussian noise
The optional vector `vars` contains descriptive names for `x` and `y`

![noisy_poly](https://cloud.githubusercontent.com/assets/10854026/13020955/9628c120-d1d8-11e5-91f3-c16367de5aad.png)

## References

- Fisher, Ronald A. "The use of multiple measurements in taxonomic problems." Annals of eugenics 7.2 (1936): 179-188.

