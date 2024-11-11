library(ggplot2)
library(dplyr)
library(cowplot)
library(stringr)
library(viridis)
library(forcats)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
    stop("At least one argument must be supplied (input file).n", call. = FALSE)
}

dir <- normalizePath(args[1])
outdir <- normalizePath(args[2])

# Function to quickly load chunks based on name
smart_load_name <- function(dir, key) {
    allfiles <- list.files(dir, pattern = ".RData")
    allfiles <- allfiles[stringr::str_detect(allfiles, key)]
    allfiles <- allfiles[which.min(stringr::str_length(allfiles))]
    filename <- stringr::str_match(allfiles, "(.+)\\.RData")[, 2]
    return(paste(dir, filename, sep = "/"))
}
scientific_10 <- function(x) {
    parse(text = gsub("e", " %*% 10^", scales::scientific_format()(x)))
}

##############################
## Predictions figure (slim)
##############################

lazyLoad(smart_load_name(dir, "predictions_by_date"))

theme_set(theme_cowplot(font_size = 9))

outputs <- c(
    paste(outdir, "/figurePredictions_slim.png", sep = ""),
    paste(outdir, "/figurePrediction_slim.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_supertile,
        ncol = 1,
        nrow = 1,
        base_height = 6,
        base_width = 14
    )
})


# ##########################
# ## Predictions by phase
# ##########################

lazyLoad(smart_load_name(dir, "predictions_by_phase"))

theme_set(theme_cowplot(font_size = 9))

outputs <- c(
    paste(outdir, "/figurePredictions_byphase.png", sep = ""),
    paste(outdir, "/figurePredictions_byphase.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_predictions_by_phase,
        scale = 0.95,
        ncol = 1,
        nrow = 1,
        base_height = 6,
        base_width = 6
    )
})

# ##########################
# ## Predictions by TA
# ##########################

lazyLoad(smart_load_name(dir, "predictions_by_therapy_summary"))

theme_set(theme_cowplot(font_size = 9))

outputs <- c(
    paste(outdir, "/figurePredictions_byTA.png", sep = ""),
    paste(outdir, "/figurePredictions_byTA.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_byta_highlights,
        scale = 0.95,
        ncol = 1,
        nrow = 1,
        base_height = 6,
        base_width = 6
    )
})

####################
## Efficacy figure
####################

lazyLoad(smart_load_name(dir, "efficacy_main"))

theme_set(theme_cowplot(font_size = 9))

p_efficacy <- plot_grid(
    p_efficacy_main,
    p_animal_main,
    rel_widths = c(1, 0.8),
    nrow = 1,
    labels = c("a", "b")
)
p_efficacy


outputs <- c(
    paste(outdir, "/figureEfficacy.png", sep = ""),
    paste(outdir, "/figureEfficacy.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_efficacy,
        scale = 0.95,
        ncol = 1,
        nrow = 1,
        base_height = 7,
        base_width = 13
    )
})

####################
## Safety figure
####################

lazyLoad(smart_load_name(dir, "safety_main"))

theme_set(theme_cowplot(font_size = 9))

p_safety <- p_safety_main

outputs <- c(
    paste(outdir, "/figureSafety.png", sep = ""),
    paste(outdir, "/figureSafety.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_safety,
        scale = 0.95,
        ncol = 1,
        nrow = 1,
        base_height = 7,
        base_width = 9
    )
})


#####################################
## Supplementary figure: dendrogram
#####################################

lazyLoad(smart_load_name(dir, "simDendroPlot"))

theme_set(theme_cowplot(font_size = 9))

p_dend <- plot_grid(
    p_dendro,
    p_heatmap,
    p_categories,
    rel_widths = c(1, 0.2, 0.6),
    nrow = 1
)
p_dend

outputs <- c(
    paste(outdir, "/figureDendro.png", sep = ""),
    paste(outdir, "/figureDendro.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_dend,
        scale = 0.95,
        ncol = 1,
        nrow = 1,
        base_height = 9,
        base_width = 7
    )
})


#####################################
## Supplementary figure: efficacy by stop reason
#####################################

lazyLoad(smart_load_name(dir, "efficacy_meta_stop_by_datatype_granular"))

theme_set(theme_cowplot(font_size = 9))

p_meta_stop_by_dt_granular

outputs <- c(
    paste(outdir, "/efficacy_byStopReason.png", sep = ""),
    paste(outdir, "/efficacy_byStopReason.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_meta_stop_by_dt_granular,
        scale = 0.95,
        ncol = 1,
        nrow = 1,
        base_height = 11,
        base_width = 8
    )
})

#####################################
## Supplementary figure: efficacy by datatource
#####################################

lazyLoad(smart_load_name(dir, "efficacy_meta_by_genetic_ds"))

theme_set(theme_cowplot(font_size = 9))

p_efficacy_meta_by_genetic_ds

outputs <- c(
    paste(outdir, "/efficacy_byGeneticDatasource.png", sep = ""),
    paste(outdir, "/efficacy_byGeneticDatasource.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_efficacy_meta_by_genetic_ds,
        scale = 0.95,
        ncol = 1,
        nrow = 1,
        base_height = 15,
        base_width = 9
    )
})

#####################################
## Supplementary figure: efficacy by L2G
#####################################

lazyLoad(smart_load_name(dir, "efficacy_gwasL2Gscore"))

theme_set(theme_cowplot(font_size = 9))

p_efficacy_gwas_l2g_score

outputs <- c(
    paste(outdir, "/efficacy_l2g.png", sep = ""),
    paste(outdir, "/efficacy_l2g.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_efficacy_gwas_l2g_score,
        scale = 0.95,
        ncol = 1,
        nrow = 1,
        base_height = 6,
        base_width = 6
    )
})

#####################################
## Supplementary figure: somatic safety
#####################################

lazyLoad(smart_load_name(dir, "safety_by_cancer_datasource"))

theme_set(theme_cowplot(font_size = 9))

p_safety_by_cancer_datasource

outputs <- c(
    paste(outdir, "/efficacy_somatic.png", sep = ""),
    paste(outdir, "/efficacy_somatic.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_safety_by_cancer_datasource,
        scale = 0.95,
        ncol = 1,
        nrow = 1,
        base_height = 6,
        base_width = 6
    )
})

#####################################
## Supplementary figure: efficacy by indication
#####################################

lazyLoad(smart_load_name(dir, "efficacy_by_indication"))

theme_set(theme_cowplot(font_size = 9))

p_efficacy_byindication

outputs <- c(
    paste(outdir, "/efficacy_byindication.png", sep = ""),
    paste(outdir, "/efficacy_byindication.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_efficacy_byindication,
        scale = 0.95,
        ncol = 1,
        nrow = 1,
        base_height = 9,
        base_width = 7
    )
})


#####################################
## Safety by indication
#####################################

lazyLoad(smart_load_name(dir, "safety_all_vs_oncology"))

theme_set(theme_cowplot(font_size = 9))

p_safety_byindication

outputs <- c(
    paste(outdir, "/safety_byindication.png", sep = ""),
    paste(outdir, "/safety_byindication.pdf", sep = "")
)

lapply(outputs, function(x) {
    save_plot(
        filename = x,
        plot = p_safety_byindication,
        scale = 0.95,
        ncol = 1,
        nrow = 1,
        base_height = 10,
        base_width = 18
    )
})
