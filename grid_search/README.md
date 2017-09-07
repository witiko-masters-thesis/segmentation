This directory contains a grid search script `grid_search.sh` that was used to
find the best BM25 and tf-idf pivoted document normalization parameters on the
dev dataset. The Jupyter notebook `grid_search.ipynb` was used to produce
a visualization of the tf-idf pivoted document normalization slope.

To reproduce the results, install the required Python packages in a Python 3
virtualenv:

    $ pip install -U pip
    $ pip install -r ../requirements.txt

download the SemEval-2016/2017 Task 3 datasets:

    $ make -C ../datasets

and run the main script:

    $ ./grid_search.sh

The results will reside in a comma-separated file named `results.csv`.
