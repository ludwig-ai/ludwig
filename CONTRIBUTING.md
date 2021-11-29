# Contributing

Everyone is welcome to contribute, and we value everybody’s contribution. Code is thus not the only
way to help the community. Answering questions, helping others, reaching out and improving the
documentation are immensely valuable contributions as well.

It also helps us if you spread the word: reference the library from blog posts on the awesome
projects it made possible, shout out on Twitter every time it has helped you, or simply star the
repo to say “thank you”.

Check out the official [ludwig docs](https://ludwig-ai.github.io/ludwig-docs/) to get oriented
around the codebase, and join the community!

## Open Issues

There is a list of open Issues at: <https://github.com/ludwig-ai/ludwig/issues>

If you would like to work on any of the open Issues:

Make sure it is not already assigned to someone else. You have the assignee (if any) on the top of
the right column of the Issue page.

You can self-assign it by commenting on the Issue page with one of the keywords: `#take` or
`#self-assign`.

Work on your self-assigned issue and eventually create a Pull Request.

## Creating Pull Requests

1. Fork the [repository](https://github.com/ludwig-ai/ludwig) by clicking on the 'Fork' button on
the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

    ```bash
    git clone git@github.com:<your Github handle>/ludwig.git
    cd ludwig
    git remote add upstream https://github.com/ludwig-ai/ludwig.git
    ```

3. Create a new branch to hold your development changes:

    ```bash
    git checkout -b a-descriptive-name-for-my-changes
    ```

    *Do not** work on the `master` branch.

4. Set up a development environment by running the following command in a virtual environment:

    ```bash
    pip install -e ".[dev]"
    pip install pre-commit
    pre-commit install
    ```

5. Develop features on your branch.

6. Format your code by running pre-commits so that your newly added files look nice:

    ```bash
    pre-commit run
    ```

    Pre-commits also run automatically when committing.

7. Once you're happy with your changes, make a commit to record your changes locally:

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

8. Once you are satisfied, go the webpage of your fork on GitHub. Click on "Pull request" to send
your to the project maintainers for review.

## Other tips

- **Add tests!** - Patches are a lot less compelling without tests.

- **Document any change in behavior** - Make sure the `README.md` and any other relevant
documentation are kept up-to-date.

- **One pull request per feature** - If you want to do more than one thing, send multiple pull
requests.

- **Send coherent history** - If you had to make multiple intermediate commits while developing,
consider [squashing them](https://www.git-scm.com/book/en/v2/Git-Tools-Rewriting-History#Changing-Multiple-Commit-Messages) before submitting.

**Happy coding**!

## Attribution

This contributing guideline is adapted from `huggingface`, available at <https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md>.
