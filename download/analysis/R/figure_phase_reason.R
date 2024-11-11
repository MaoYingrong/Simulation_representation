library(ggplot2)
library(dplyr)
library(forcats)
library(cowplot)
library(stringr)

# Load data
data <- read.csv("./data/reason_vs_phase_predictions.csv")

plot <- data %>%
    filter(!(comparison %in% (c("Success", "Insufficient_Data", "No_Context", "Interim_Analysis", "Another_Study")))) %>%
    filter(prediction != "Other") %>%
    mutate(is_significant = pvalue < 0.05) %>%
    mutate(comparison = str_replace_all(comparison, "_", " ")) %>%
    mutate(comparison = fct_rev(fct_reorder(comparison, comparisonTotal))) %>%
    mutate(prediction = fct_rev(fct_relevel(prediction, c("Early Phase 1", "Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3", "Phase 3", "Phase 4")))) %>%
    ggplot(aes(x = or_result, y = prediction, color = is_significant)) +
    geom_vline(aes(xintercept = 1),
        linewidth = .25,
        linetype = "dashed"
    ) +
    geom_errorbar(
        aes(xmin = lower_ci, xmax = upper_ci),
        width = 0,
        # color = "steelblue",
        linewidth = 1
    ) +
    geom_point(
        # color = "steelblue",
        size = 3
    ) +
    scale_color_manual(
        name = "P-value",
        values = c("darkgrey", "steelblue"),
        labels = c(">0.05", "<0.05")
    ) +
    scale_x_log10(name = "Odds Ratio") +
    facet_wrap(. ~ comparison, ncol = 5) +
    theme_cowplot(font_size = 12) +
    panel_border(color = "black") +
    theme(
        axis.title.y = element_blank(),
        axis.line = element_blank(),
        panel.spacing = unit(-0.1, "lines"),
        legend.position = "bottom",
        strip.placement = "outside",
        strip.background = element_blank(),
    )
ggsave("./temp/figures/test_figure_phase_reason.png", plot, width = 9, height = 6, dpi = 300)
