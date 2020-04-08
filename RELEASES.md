Releasing
=========

Release procedure
-----------------

1. Update version number in `ludwig/globals.py`
2. Update version number in `setup.py`
3. Commit
4. Tag the commit with the version number `vX.Y.Z` with a meaningful message
5. Push with `--tags`
6. If a non-patch release, edit the release notes
7. Create a release for Pypi: `python setup.py sdist`
8. Release on Pypi: `twine upaload --repository pypi dist/ludwig-X.Y.Z.tar.gz`

Release policy
--------------

Ludwig follows [Semantic Versioning](https://semver.org).
In general, for major and minor releases, maintainers should all agree on the release.
For patches, in particular time sensitive ones, a single maintainer can release without a full consensus, but this practice should be reserved for critical situations.
