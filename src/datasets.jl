"""
`X, y, vars = load_iris(n)`

Loads the first `n` (of 150) observations from the
Iris flower data set introduced by Ronald Fisher (1936).
The 4 by `n` matrix `X` contains the numeric measurements,
in which each individual column denotes an observation.
The vector `y` contains the class labels as strings.
The optional vector `vars` contains the names of the features (i.e. rows of `X`)

[1] Fisher, Ronald A. "The use of multiple measurements in taxonomic problems." Annals of eugenics 7.2 (1936): 179-188.
"""
function load_iris(n::Int = 150)
    @assert 0 < n <= 150
    path = joinpath(Pkg.dir("MLDataUtils"), "data", "iris.csv")
    raw_csv = readcsv(path)
    X = convert(Matrix{Float64}, raw_csv[1:n, 1:4]')
    y = convert(Vector{ASCIIString}, raw_csv[1:n, 5])
    vars = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    X, y, vars
end

"""
`x, y, vars = load_sin()`

Loads an artificial example dataset for a noisy sin.
It is particularly useful to explain under- and overfitting.
The vector `x` contains equally spaced points between 0 and 2π
The vector `y` contains `sin(x)` plus some gaussian noise
The optional vector `vars` contains descriptive names for `x` and `y`
"""
function load_sin()
    path = joinpath(Pkg.dir("MLDataUtils"), "data", "sin.csv")
    raw_csv = readcsv(path)
    x = convert(Vector{Float64}, raw_csv[:, 1])
    y = convert(Vector{Float64}, raw_csv[:, 2])
    x, y, ["X", "sin(X) + ɛ"]
end

"""
`x, y, vars = load_line()`

Loads an artificial example dataset for a noisy line.
It is particularly useful to explain under- and overfitting.
The vector `x` contains 11 equally spaced points between 0 and 1
The vector `y` contains `x ./ 2 + 1` plus some gaussian noise
The optional vector `vars` contains descriptive names for `x` and `y`
"""
function load_line()
    path = joinpath(Pkg.dir("MLDataUtils"), "data", "line.csv")
    raw_csv = readcsv(path)
    x = convert(Vector{Float64}, raw_csv[:, 1])
    y = convert(Vector{Float64}, raw_csv[:, 2])
    x, y, ["x", "0.5 x + 1 + ɛ"]
end

"""
`x, y, vars = load_poly()`

Loads an artificial example dataset for a noisy quadratic function.
It is particularly useful to explain under- and overfitting.
The vector `x` contains 50 points between 0 and 4
The vector `y` contains `2.6 * x^2 + .8 * x` plus some gaussian noise
The optional vector `vars` contains descriptive names for `x` and `y`
"""
function load_poly()
    path = joinpath(Pkg.dir("MLDataUtils"), "data", "poly.csv")
    raw_csv = readcsv(path)
    x = convert(Vector{Float64}, raw_csv[:, 1])
    y = convert(Vector{Float64}, raw_csv[:, 2])
    x, y, ["x", "2.6 x² + .8 x + ɛ"]
end
