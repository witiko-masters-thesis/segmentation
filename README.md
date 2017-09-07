# Requirements

To run the code, you will require:

- Python 2 and Python 3,
- Info-ZIP `unzip` 6.0,
- GNU Bash, Make, Parallel, Wget, and
- `sort` from GNU coreutils.

# Usage

To reproduce the results, install the required Python packages in a Python 3
virtualenv:

    $ pip install -U pip
    $ pip install -r requirements.txt

download the SemEval-2016/2017 Task 3 datasets:

    $ make -C datasets

and run the main script:

    $ ./__main__.sh

The results will reside in three comma-separated files named
`results-dev.csv`, `results-2016.csv`, and `results-2017.csv`.
