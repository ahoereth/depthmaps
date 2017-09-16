all:
	pandoc meta.yaml [1-9][0-9]_*.md \
		-F pandoc-crossref \
		-F pandoc-citeproc \
		--bibliography library.bib \
		-o report.pdf
