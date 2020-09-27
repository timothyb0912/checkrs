# checkrs

Tools for simulation-based model checking.

## Description

The checkrs package contains functions for creating 7 model checking/diagnostic plots described in
> Brathwaite, Timothy. "Check yourself before you wreck yourself: Assessing
discrete choice models through predictive simulations" arXiv preprint
arXiv:1806.02307 (2018). https://arxiv.org/abs/1806.02307.

Beyond the plots described in this paper, checkrs enables the creation of reliability and marginal model plots that use continuous scatterplot smooths based on [Extremely Randomized Trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html#sklearn.tree.ExtraTreeClassifier) as kernel estimators, as opposed to only allowing discrete smooths based on binning.

As for the name, checkrs is a play on the word "checkers," i.e., those tools one uses to check, or one who checks.
The name is also a play on the phrases "check the research of scientists" and "check research scientists."

## Usage Installation

`pip install checkrs`

## To-Do:
   - Add usage examples
   - Add tests
   - Set up tox
   - Set up pre-commit
   - Set up continuous integration
   - Refactor to remove pandas dependency
   - Architecture overhaul to go from prototype to v1.
   - Add package to conda and conda-forge

## Development installation

To work on and edit checkrs, the following setup process may be useful.

1. from the project root, create an environment `checkrs` with the help of [conda](https://docs.conda.io/en/latest/),
   ```
   cd checkrs
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```
   conda activate checkrs
   ```
3. install `checkrs` in an editable fashion using:
   ```
   pip install -e .
   ```

Optional and needed only once after `git clone`:

4. install several [pre-commit] git hooks with:
   ```
   pre-commit install
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

5. install [jupytext] git hooks to store notebooks as formatted python files:
   ```
   #!/bin/sh
   # For every ipynb file in the git index:
   # - apply black and flake8
   # - export the notebook to a Python script in folder 'python'
   # - and add it to the git index
   jupytext --from ipynb --pipe black --check flake8 --pre-commit
   jupytext --from ipynb --to py:light --pre-commit
   ```
   This is useful to avoid large diffs due to plots in your notebooks.

Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yaml`, `requirements.in`, and eventually
in `setup.cfg` and  if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yaml` and `requirements.txt` for the exact reproduction of your
   environment with:
   ```
   pip-compile requirements.in
   conda env export -n checkrs -f environment.lock.yaml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yaml` using:
   ```
   conda env update -f environment.lock.yaml --prune
   ```
   Or
   ```
   pip install -r requirements.txt
   ```

## Project Organization

```
├── AUTHORS.rst             <- List of developers and maintainers.
├── CHANGELOG.rst           <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yaml        <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── checkrs             <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## Note

This project has been set up using PyScaffold 3.3a1 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
