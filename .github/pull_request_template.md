# Code Pull Requests

Please provide the following:

- a clear explanation of what your code does
- if applicable, a reference to an issue
- a reproducible test for your PR (code, config and data sample)

# Documentation Pull Requests

Note that the documentation HTML files are in `docs/` while the Markdown sources are in `mkdocs/docs`.

If you are proposing a modification to the documentation you should change only the Markdown files.

`api.md` is automatically generated from the docstrings in the code, so if you want to change something in that file, first modify `ludwig/api.py` docstring, then run `mkdocs/code_docs_autogen.py`, which will create `mkdocs/docs/api.md` .
