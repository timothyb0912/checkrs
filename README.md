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

## Installation

`pip install checkrs`

## Usage
Note that `example_project` is fictitious! This example is, literally, just an example.
```
from checkrs import ChartData, ViewSimCDF

from example_project import load_data

design, targets_observed, targets_simulated = load_data()

chart_data = ChartData.from_raw(
  targets=targets_observed,  # 1D Ndarray or Tensor
  targets_simulated=targets_simulated, # 2D Ndarray or Tensor
  design=design # DataFrame or None
)

chart = ViewSimCDF.from_chart_data(chart_data)

chart_plotnine = chart.draw(backend="plotnine")
chart_altair = chart.draw(backend="altair")

####
## Save to a variety of formats
####
# chart.save("temp_plot.png")
# chart.save("temp_plot.pdf")
# chart.save("temp_plot.json")
# chart.save("temp_plot.html")
```
See docstrings for `ChartData.from_raw`, `ViewSimCDF.from_chart_data`, and `ViewSimCDF.save`.

## To-Do:
   - Set up pre-commit
   - Add package to conda and conda-forge

## Development installation

To work on and edit checkrs, the following setup process may be useful.

1. from the project root, create an environment `checkrs` with the help of [conda](https://docs.conda.io/en/latest/),
   ```
   cd checkrs
   conda env create -n checkrs -f environment.yml
   ```
2. activate the new environment with
   ```
   conda activate checkrs
   ```
3. install `checkrs` in an editable fashion using:
   ```
   flit install --pth-file
   ```

Optional and needed only once after `git clone`:

4. install several [pre-commit] git hooks with:
   ```
   pre-commit install
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml`, `requirements.in`, and eventually in `pyproject.toml` if you want to ship and install the package via `pip` later on.

   Use `environment.yml` for dependencies that cannot be installed via `pip`.
   Use `requirements.in` for dependencies that can be installed via `pip`.
   Use `pyproject.toml` for dependencies that are needed for `checkrs` to function at all, not just in development.
2. Create concrete dependencies as `requirements.txt` for the exact reproduction of your environment with:
   ```
   pip-compile requirements.in
   ```
3. Manually update any non-pip dependencies in `environment.yml`, being sure to pin any such dependencies to a specific version.
3. Update your current environment using:
   ```
   conda env update -f environment.yml
   ```
   Or
   ```
   pip install -r requirements.txt
   ```
   if you did not update any non-pip dependencies.

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
