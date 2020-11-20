=========
Changelog
=========

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

.. towncrier release notes start

Checkrs 0.2.0 (2020-11-19)
==========================

Added new features
------------------

- Exposed data attribute on View objects. (#40)


Changed existing functionality
------------------------------

- Implemented keepachangelog format for CHANGELOG.rst (#37)


Bug fixes
---------

- Removed incorrect requirements.txt flag. (#43)


Checkrs 0.1.2 (2020-11-13)
==========================

Added new features
------------------

- Added declarative plotting objects: ViewSimCDF, ChartData, and View. (#11)
- Added tox for cross-version testing. (#14)
- Updated README and added jupytext + {black, flake8} to pre-commit. (#16)
- Added classifier tags for supported python versions to pyproject.toml (#18)
- Added README badge for Github-Actions Tests workflow. (#21)


Bug fixes
---------

- Fixed incorrect package requirements in pyproject.toml. (#11)


Removed from package
--------------------

- Removed private method that is no longer needed. (#12)


Checkrs 0.1.1 (2020-09-27)
==========================

Added new features
------------------

- Moved project build instructions from setup.cfg to pyproject.toml.
  Moved project publishing from pyscaffold to flit. (#6)


Checkrs 0.1.0 (2020-09-27)
==========================

Added new features
------------------

- Uploaded the initial package version to PyPI.
- Set up package development requirements files. (#1)
