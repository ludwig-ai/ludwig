# Code Pull Requests

Please provide the following:

- a clear explanation of what your code dose
- if applicable, a reference to an issue
- a reproducible test for your PR (code, model definition and data sample)

# Documentation Pull Requests

Note that the documentation HTML files are a in `docs/` while the Markdown sources are in `mkdocs/docs`.

If you are proposing a modification to the documentation you should change only the Markdown files, then recreate the documentation as examplained in the `mkdocs/README.md` file with `mkdocs build`, which will create the HTML files automatically, and only after this create a commit.

`API.md` is automatically generated from the docstrings in the code, so if you want to change something in that file, first modify `ludwig/api.py` docstring, then run `mkdocs/code_docs_autogen.py`, which will create `mkdocs/docs/api.md` and then finally run `mkdocs build` which will generate the HTML in `docs/`.
