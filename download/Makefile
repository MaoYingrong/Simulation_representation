# Makefile ---

# Author: David Ochoa <ochoa@ebi.ac.uk>

ROOTDIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

RANALYSISDIR= $(ROOTDIR)/analysis/R
TMPDIR = $(ROOTDIR)/temp/
DOCSDIR= $(TMPDIR)/docs

FIGSDIR= $(TMPDIR)/figures
FIGSDIRONCO= $(TMPDIR)/figures_oncology
FIGSDIRNONONCO= $(TMPDIR)/figures_nononcology

#Reports
REPORT = $(DOCSDIR)/index.html
REPORTONCO = $(DOCSDIR)/index_oncology.html
REPORTNONONCO = $(DOCSDIR)/index_nononcoology.html

# Programs
# R ?= $(shell which R)
# RSCRIPT ?= $(shell which Rscript)

figures: $(FIGSDIR)
	poetry run Rscript $(RANALYSISDIR)/figures.R './temp/report_cache/' $(FIGSDIR)
	
clean-report:
	-rm $(REPORT)

clean-report-all:
	-rm -rf $(TMPDIR)

setup-dev: SHELL:=/bin/bash
setup-dev: ## Setup development environment
	@. utils/install_dependencies.sh

report: $(REPORT)

$(TMPDIR):
	mkdir -p $@

$(DOCSDIR):
	mkdir -p $@

$(FIGSDIR):
	mkdir -p $@

$(FIGSDIRONCO):
	mkdir -p $@

$(FIGSDIRNONONCO):
	mkdir -p $@

#Report
$(REPORT): $(TMPDIR) $(DOCSDIR) $(FIGSDIR)
	poetry run R \
	-e "cache_dir <- '$(TMPDIR)/report_cache/'" \
	-e "config_path <- '$(ROOTDIR)/config.yml'" \
	-e "rmarkdown::render('$(RANALYSISDIR)/report.rmd', knit_root_dir='$(ROOTDIR)',output_file = '$@')"

$(REPORTONCO): $(TMPDIR) $(DOCSDIR) $(FIGSDIRONCO)
	poetry run R \
	-e "cache_dir <- '$(TMPDIR)/report_cache_oncology/'" \
	-e "config_path <- '$(ROOTDIR)/config_oncology.yml'" \
	-e "rmarkdown::render('$(RANALYSISDIR)/report.rmd', knit_root_dir='$(ROOTDIR)',output_file = '$@')"

$(REPORTNONONCO): $(TMPDIR) $(DOCSDIR) $(FIGSDIRNONONCO)
	poetry run R \
	-e "cache_dir <- '$(TMPDIR)/report_cache_nononcology/'" \
	-e "config_path <- '$(ROOTDIR)/config_nononcology.yml'" \
	-e "rmarkdown::render('$(RANALYSISDIR)/report.rmd', knit_root_dir='$(ROOTDIR)',output_file = '$@')"