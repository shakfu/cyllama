VERSION := "0.1.16"

.PHONY: all update build pdf clean

all: build

build: pdf
	@echo "Update and PDF build complete."

pdf:
	@quarto render --to pdf -o "cyllama-userguide-$(VERSION).pdf"

clean:
	@rm -rf .quarto _book
	@find . \( -name '*.aux' -o -name '*.log' -o -name '*.pdf' -o -name '*.tex' -o -name '*.toc' \) -delete
