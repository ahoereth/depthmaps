pdf: article.pdf clean
ps: article.ps clean

%.pdf: %.ps
	ps2pdf $<

%.ps: %.tex
	latex $<
	bibtex $*
	latex $<
	latex $<
	dvips $*

%.tex: index.md
	pandoc \
		--bibliography library \
		--template esann.template \
		--number-sections \
		--filter pandoc-crossref \
		--natbib \
		-o $@ \
		meta.yaml index.md

clean_all: clean
	rm -f *.pdf *.ps

clean:
	rm -f *.aux *.bbl *.bcf *.blg *.dvi *.fdb_latexmk *.fls *.log *.out *.run.xml *.tex
	rm -f *.ps
