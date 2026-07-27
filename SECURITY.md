# Security Policy

## Supported Versions

Security fixes are released against the latest patch version of the current
minor release. We do not backport fixes to older minor versions.

| Version | Supported          |
| ------- | ------------------ |
| 0.17.x  | :white_check_mark: |
| < 0.17  | :x:                |

If you are running an older version, please upgrade before reporting an issue,
so we can confirm it still reproduces on a supported release.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
pull requests, or Discord.**

Report privately through GitHub Private Vulnerability Reporting:

- Go to [Report a vulnerability](https://github.com/ludwig-ai/ludwig/security/advisories/new), or
- Open the **Security** tab of this repository and click **Report a vulnerability**

This creates a private advisory that only the reporter and Ludwig maintainers
can see, so we can coordinate a fix before anything is disclosed publicly.

To help us triage quickly, please include as much of the following as you can:

- The Ludwig version, Python version, and operating system
- The affected component (for example: preprocessing, a specific encoder, the
  serving API, a backend such as Ray)
- A description of the vulnerability and the impact you believe it has
- Steps to reproduce, ideally a minimal proof of concept
- Any suggested fix or mitigation you have in mind

## What to Expect

Ludwig is maintained by a small group of contributors, so these are good faith
targets rather than a contractual SLA:

- **Acknowledgement:** within 5 business days
- **Initial assessment:** within 10 business days, including whether we consider
  the report in scope and a rough severity
- **Fix and disclosure:** we will keep you updated as we work on a fix, publish
  a GitHub Security Advisory when it ships, and request a CVE where appropriate

We are happy to credit you in the advisory. Let us know how you would like to be
named, or tell us if you would rather stay anonymous.

Please give us a reasonable window to ship a fix before disclosing publicly. If
you have a disclosure deadline you need to meet, say so up front and we will
work with it.

## Scope and Trust Model

Ludwig is a machine learning framework. Like most of the Python ML ecosystem, it
assumes that the artifacts and configuration you hand it are trusted inputs. The
following are known and expected behaviors rather than vulnerabilities:

- **Loading models and checkpoints from untrusted sources can execute arbitrary
  code.** Ludwig checkpoint files carry training state that is deserialized with
  `torch.load(..., weights_only=False)`, and some cached artifacts are read with
  `pickle`. Only load models you produced yourself or obtained from a source you
  trust. Model weights themselves are loaded with `weights_only=True`.
- **Ludwig configs and dataset paths are executable surface.** A config controls
  file paths, remote URLs, and which model code runs. Treat a config from an
  untrusted party the same way you would treat a script from that party.
- **Downloaded datasets and pretrained encoders** come from third party sources
  (for example Hugging Face Hub, Kaggle). Their contents are outside Ludwig's
  control.

Reports that are in scope include things like: a bug that lets a crafted
*dataset file* (as opposed to a model artifact) achieve code execution during
preprocessing, path traversal when extracting archives, injection in a
Ludwig-provided serving endpoint, or a dependency vulnerability that Ludwig
actually exposes in a reachable way.

If you are not sure whether something is in scope, report it anyway and we will
work it out together.

## Non-Security Bugs

For regular bugs, please use the [issue tracker](https://github.com/ludwig-ai/ludwig/issues).
For questions and general discussion, join us on
[Discord](https://discord.gg/CBgdrGnZjy).
