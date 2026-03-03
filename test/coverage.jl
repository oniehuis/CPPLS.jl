using Pkg
Pkg.instantiate()
using Coverage

coverage = vcat(
    process_folder("src"),
    process_folder("ext"),
)

LCOV.writefile("lcov.info", coverage)
