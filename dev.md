Blurb taken from [HEIR](https://heir.dev/docs/development/#pre-commit)

## Pre-Commit

We use [pre-commit](https://pre-commit.com/) to manage a series of git
pre-commit hooks for the project; for example, each time you commit code, the
hooks will make sure that your C++ is formatted properly. If your code isn't,
the hook will format it, so when you try to commit the second time you'll get
past the hook.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks, run

```bash
pip install -r requirements-dev.txt
```

Then install the hooks to run automatically on `git commit`:

```bash
pre-commit install
```

To run them manually, run

```bash
pre-commit run --all-files
```
