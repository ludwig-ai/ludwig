# Releasing

## Release procedure

1. Update version number in `ludwig/globals.py`
1. Update the `README.md` file
1. Update `ludwig-docs`
1. Commit
1. Tag the commit with the version number `vX.Y.Z` with a meaningful message
1. Push with `--tags`
1. If a non-patch release, edit the release notes
1. The PyPI upload is automated via GitHub Actions (`.github/workflows/upload-pypi.yml`) when a release is published
1. Publish Docker images (see below)

## Docker images

Four images are published to Docker Hub under the `ludwigai` organisation for each release:
`ludwigai/ludwig`, `ludwigai/ludwig-gpu`, `ludwigai/ludwig-ray`, `ludwigai/ludwig-ray-gpu`.

### Automated (CI)

The GitHub Actions workflow `.github/workflows/docker.yml` triggers on `v*.*.*` tags and builds
images from the tagged source. If CI is healthy this runs automatically after step 6 above.

### Manual fallback

If CI does not run or images need to be backfilled, use the script at `docker/build_and_push.sh`.
You must be logged in to Docker Hub as a member of the `ludwigai` organisation first.

```bash
# Check login (docker login if needed)
docker login

# Build and push all 4 image variants for a release.
# Add --latest for the newest release.
./docker/build_and_push.sh 0.14.0 --latest
```

This installs `ludwig[full]==<version>` from PyPI into the images and pushes two tags per image:
the full version (`0.14.0`) and the major.minor shorthand (`0.14`), plus `latest` when requested.

## Release policy

Ludwig follows [Semantic Versioning](https://semver.org).
In general, for major and minor releases, maintainers should all agree on the release.
For patches, in particular time sensitive ones, a single maintainer can release without a full consensus, but this practice should be reserved for critical situations.
