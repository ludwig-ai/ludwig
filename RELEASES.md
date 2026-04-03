# Releasing

## Release procedure

1. Update version number in `ludwig/globals.py`
1. Commit
1. Tag the commit with the version number `vX.Y.Z` with a meaningful message
1. Push with `--tags`
1. If a non-patch release, edit the release notes
1. The PyPI upload is automated via GitHub Actions (`.github/workflows/upload-pypi.yml`) when a release is published

## Release policy

Ludwig follows [Semantic Versioning](https://semver.org).
In general, for major and minor releases, maintainers should all agree on the release.
For patches, in particular time sensitive ones, a single maintainer can release without a full consensus, but this practice should be reserved for critical situations.
