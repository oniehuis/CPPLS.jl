using Documenter
using Markdown
using Makie
using CPPLS

function ensure_makie_extension!()
    ext = Base.get_extension(CPPLS, :MakieExtension)
    return ext !== nothing ? ext :
           Base.require_extension(CPPLS, :MakieExtension)
end

ensure_makie_extension!()

DocMeta.setdocmeta!(
    CPPLS,
    :DocTestSetup,
    :(using CPPLS);
    recursive = true,
)

makedocs(
    sitename = "CPPLS",
    format = Documenter.HTML(mathengine = Documenter.MathJax()),
    modules = [CPPLS],
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

deploydocs(repo = "github.com/oniehuis/CPPLS.jl", devbranch = "main")
