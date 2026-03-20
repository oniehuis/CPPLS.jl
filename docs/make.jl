import Pkg

if Base.active_project() ≠ joinpath(@__DIR__, "Project.toml")
    Pkg.activate(@__DIR__)
end

strict_docs_env = get(ENV, "CPPLS_DOCS_STRICT_ENV", "false") == "true"

if strict_docs_env
    empty!(LOAD_PATH)
    append!(LOAD_PATH, ["@", "@stdlib"])
end

if get(ENV, "CI", "false") == "true"
    Pkg.instantiate()
end

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
    format = Documenter.HTML(mathengine = Documenter.MathJax2()),
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
            "CPPLS/visualization.md"
        ],
        "Utils" => Any[
            "Utils/encoding.md",
            "Utils/statistics.md",
            "Utils/crossvalidation_utils.md"
        ],
        "Register" => "register.md",
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
