# Contributing

Everyone is welcome to contribute, and we value everybody’s contribution. Code is thus not the only
way to help the community. Answering questions, helping others, reaching out and improving the
documentation are immensely valuable contributions as well.

It also helps us if you spread the word: reference the library from blog posts on the awesome
projects it made possible, shout out on Twitter every time it has helped you, or simply star the
repo to say "thank you".

Check out the official [ludwig docs](https://ludwig-ai.github.io/ludwig-docs/) to get oriented
around the codebase, and join the community!

## Open Issues

Issues are listed at: <https://github.com/ludwig-ai/ludwig/issues>

If you would like to work on any of them, make sure it is not already assigned to someone else.

You can self-assign it by commenting on the Issue page with one of the keywords: `#take` or
`#self-assign`.

Work on your self-assigned issue and eventually create a Pull Request.

## Creating Pull Requests

1. Fork the [repository](https://github.com/ludwig-ai/ludwig) by clicking on the 'Fork' button on
   the repository's page. This creates a copy of the code under your GitHub user account.

1. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your Github handle>/ludwig.git
   cd ludwig
   git remote add upstream https://github.com/ludwig-ai/ludwig.git
   ```

1. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   *Do not*\* work on the `master` branch.

1. Set up a development environment by running the following command in a virtual environment:

   ```bash
   pip install -e ".[dev]"
   pip install pre-commit
   pre-commit install
   ```

1. Develop features on your branch.

1. Format your code by running pre-commits so that your newly added files look nice:

   ```bash
   pre-commit run
   ```

   Pre-commits also run automatically when committing.

1. Once you're happy with your changes, make a commit to record your changes locally:

   ```bash
   git add .
   git commit
   ```

   It is a good idea to sync your copy of the code with the original repository regularly. This
   way you can quickly account for changes:

   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

   Push the changes to your account using:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

1. Once you are satisfied, go the webpage of your fork on GitHub. Click on "Pull request" to send
   your to the project maintainers for review.

## Other tips

- Use [google/yapf](https://github.com/google/yapf) to format Python code.
- Add unit tests for any new code you write.
- Make sure tests pass. See the [Developer Guide](https://ludwig-ai.github.io/ludwig-docs/latest/developer_guide/style_guidelines_and_tests/) for more details.

## Attribution

This contributing guideline is adapted from `huggingface`, available at <https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md>.

## Code of Conduct

Please be mindful of and adhere to the Linux Foundation's
[Code of Conduct](https://lfprojects.org/policies/code-of-conduct) when contributing to Ludwig.
