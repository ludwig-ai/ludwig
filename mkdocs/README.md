Ludwig documentation
====================

Ludwig's documentation is build using [MkDocs](https://www.mkdocs.org/) and the beautiful [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.
In order to create Ludwig's documentation you have to install them:

```
pip install mkdocs mkdocs-material
```

Be sure that you installe version of `Markdown>=3.0.1`. Then generate `api.md` from source (from the `mkdocs` directory):

```
python code_doc_autogen.py
```

Test it (from the `mkdocs` directory):

```
mkdocs serve
```

Finally build the static website (from the `mkdocs` directory):

```
mkdocs build
```

It will create the static website in `$LUDWIG_HOME/docs/`.