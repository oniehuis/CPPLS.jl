using Documenter
using Markdown
using Makie
using CPPLS
using StatsAPI

DocMeta.setdocmeta!(
    CPPLS,
    :DocTestSetup,
    :(using CPPLS; using StatsAPI);
    recursive = true,
)

makedocs(
    sitename = "CPPLS",
    format = Documenter.HTML(mathengine = Documenter.MathJax()),
    modules = [CPPLS],
    checkdocs = :exports,
    authors = "Oliver Niehuis",
    pages = [
        "Home" => "index.md",
        "CPPLS" => Any[
            "CPPLS/theory.md",
            "CPPLS/types.md",
            "CPPLS/fit.md",
            "CPPLS/predict.md",
            "CPPLS/crossvalidation.md",
            "CPPLS/visualization.md",
            "CPPLS/internal.md",
        ],
        "Utils" => Any[
            "Utils/encoding.md",
            "Utils/matrix.md",
            "Utils/statistics.md",
            "Utils/internal.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/oniehuis/CPPLS.jl", 
    devbranch = "main", 
    push_preview = false,
    versions = [
		"stable" => "v^",
		"v#.#.#",
		"dev" => "dev",
	],
)
