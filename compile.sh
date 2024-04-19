#!/usr/bin/env zsh 
mkdir out
pdflatex -output-directory=out autoencoders.tex
mv out/autoencoders.pdf .
evince autoencoders.pdf &
