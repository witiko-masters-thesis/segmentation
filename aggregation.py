"""This module provides segment similarity aggregation functions."""

import logging

LOGGER = logging.getLogger(__name__)

def harmonic_number(n, s):
    """Returns the generalized harmonic number Hn,s."""
    return sum(1 / (i**s) for i in range(1, n+1))

def aggregate_min(results, _):
    """aggregate_score is the minimum score."""
    results = [result for result in results if result[-1].active]
    return min(results, key=lambda result: result[0])[:-1]

def aggregate_max(results, _):
    """aggregate_score is the maximum score."""
    results = [result for result in results if result[-1].active]
    return max(results, key=lambda result: result[0])[:-1]

def aggregate_avg(results, _):
    """aggregate_score is the average score."""
    results = [result for result in results if result[-1].active]
    average = sum(result[0] for result in results) / len(results)
    return [average] + results[0][1:-1]

def aggregate_wavg_length(results, _):
    """
        aggregate_score is the weighted average score with weights proportional
        to len(variable_nugget).
    """
    results = [result for result in results if result[-1].active]
    weights = [len(result[-1].tokens) for result in results]
    if sum(weights) == 0:
        average = 0.0
    else:
        average = sum(result[0] * (weight / sum(weights)) \
                      for result, weight in zip(results, weights))
    return [average] + results[0][1:-1]

def aggregate_wavg_godwin(results, _):
    """
        aggregate_score is the weighted average score with weights proportional
        to the inverse rank of a nugget.
    """
    s = 1
    weights = [1 / ((rank+1)**s) for rank, result in enumerate(results) \
               if result[-1].active]
    results = [result for result in results if result[-1].active]
    if sum(weights) == 0:
        average = 0.0
    else:
        average = sum(result[0] * (weight / sum(weights)) \
                      for result, weight in zip(results, weights))
    return [average] + results[0][1:-1]

def aggregate_wavg_koetal04(results, language_model):
    """
        aggregate_score is the weighted average score with weights proportional
        to the similarity of a nugget to the title of its originating document.
    """
    results = [result for result in results if result[-1].active]
    weights = [language_model.similarity(result[-1].document.qsubject, result[-1]) \
               for result in results]
    if sum(weights) == 0:
        average = 0.0
    else:
        average = sum(result[0] * (weight / sum(weights)) \
                      for result, weight in zip(results, weights))
    return [average] + results[0][1:-1]
