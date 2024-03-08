# Contributing to this repository

## Install linter

First of all, you need to install `ruff` package to verify that you passed all conditions for formatting.

```
pip install ruff==0.0.287
```

### Apply linter before PR

Please run the ruff check with the following command:

```
ruff check .
```

### Auto-fix with fixable errors

```
ruff check . --fix
```