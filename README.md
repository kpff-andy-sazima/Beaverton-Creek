# Beaverton Creek Structural Calculations

## Introduction

These calculations are all done using Python. They pull in reused tools from Andy Sazima's *Structural Tools* package, along with other packages like *handcalcs*, *forallpeople*, *pandas*, *PyNite*, and more.

## Creation

Because Python files are just text files, you can open these files in any text editor. I highly recommend VSCode or Spyder, but something as simple as Notepad would do in a pinch.

## Tool chain

The full workflow for these Python calculations is as follows.

**Environment creation**: UV/pip are used to build the virtual python environment. Simply run `uv sync` in the root directory of this repository.
**Configuration**: All configurations are kept in the `pyproject.toml` file.
**Running**: You can run python files with `python3 <file>`.
**Export**: To export to PDF, use the custom export script by calling `export-nb pdf <file>`. 

## Explanation of Python vs Jupyter and their usage in this project

A small glossary of terms:

- Python: a popular programming language.
- Markdown: a popular markup language. This is a quick way to express formatting (bold, italic, headings, etc.) in a purely text format (as opposed to something like an MS Word document).
- Jupyter notebook: a filetype that combines markdown and python into blocks for easy explanation and demonstration.

Jupyter notebooks are extremely easy to export to a report-style PDF, so they are the main way calcs are done in this project; however, they can be a bit slow in VSCode when they get large. For that reason, I like to write the Jupyter notebooks in pure python (.py), then convert them to Jupyter notebook (.ipynb) with the package `jupytext`. This is automatically done by the `export-nb` script. 

Because of this conversion, each file will have 2-3 copies: one for the python file, one for the Jupyter notebook, and one for the PDF once it's exported.
