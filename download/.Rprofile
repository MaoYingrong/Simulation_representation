Sys.setenv(TERM_PROGRAM = "vscode")
source(file.path(
    Sys.getenv(
        if (.Platform$OS.type == "windows") "USERPROFILE" else "HOME"
    ),
    ".vscode-R", "init.R"
))
source("renv/activate.R")
