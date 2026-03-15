"""
	dataset(name::AbstractString)

Return the absolute path to an example dataset file shipped with CPPLS. The argument
`name` is interpreted relative to the package's `examples` directory.

# Examples
```jldoctest
julia> isfile(CPPLS.dataset("synthetic_cppls_da_dataset.jld2"))
true
```
"""
dataset(name::AbstractString) = joinpath(pkgdir(CPPLS), "examples", name)