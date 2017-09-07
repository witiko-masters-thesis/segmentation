"""This module contains high-level training and evaluation functions."""

import logging
from sklearn.linear_model import LogisticRegression

from preprocessing import segment_threads, segment_orgquestions

LOGGER = logging.getLogger(__name__)
LOGISTIC_REGRESSION_RANDOM_STATE = 12345

def produce_gold_results(dataset_fnames, output_fname):
    """
        Produces gold results from an input (dev) datasets and stores the
        results in an output file.
    """
    with open(output_fname, "wt") as output_file:
        orgquestion_ids = []
        orgquestion_threads = {}
        for orgquestion, (thread, relevant) in zip(segment_orgquestions(dataset_fnames),
                                                   segment_threads(dataset_fnames)):
            if orgquestion.id not in orgquestion_threads:
                orgquestion_threads[orgquestion.id] = []
                orgquestion_ids.append(orgquestion.id)
            orgquestion_threads[orgquestion.id].append((relevant, thread.id))
        for orgquestion_id in orgquestion_ids:
            threads = orgquestion_threads[orgquestion_id]
            sorted_threads = sorted(enumerate(threads), key=lambda thread: thread[1][0], \
                                    reverse=True)
            for rank, (_, (relevant, thread_id)) in sorted(enumerate(sorted_threads),
                                                           key=lambda thread: thread[1][0]):
                gold_score = (len(sorted_threads)-rank)/len(sorted_threads)
                output_file.write("%s\t%s\t%d\t%s\t%s\n" % (orgquestion_id, thread_id,
                                                            rank+1, gold_score,
                                                            "true" if relevant else "false"))

def train_nonsegmented(language_model, dataset_fnames, segment_filtering=False):
    """
        Trains a classifier that maps document similarity to relevance labels.
        The non-segmented version disregards segmentation and computes similarity directly between
        documents.

        If segment_filtering is not None, a text summarization technique is
        used for the filtering of <Thread> segments.
    """
    training_scores = []
    training_classes = []
    for orgquestion, (thread, relevant) \
        in zip(segment_orgquestions(dataset_fnames),
               segment_threads(dataset_fnames, segment_filtering=segment_filtering)):
        training_scores.append([language_model.similarity(orgquestion, thread)])
        training_classes.append(relevant)
    classifier = LogisticRegression(random_state=LOGISTIC_REGRESSION_RANDOM_STATE)
    classifier.fit(training_scores, training_classes)
    return classifier

def evaluate_nonsegmented(language_model, classifier, dataset_fnames, output_fname, \
                          segment_filtering=None):
    """
        Produces an output file that contains the ranking of document pairs and
        predicted relevance labels.  The non-segmented version disregards
        segmentation and computes similarity directly between documents.

        If segment_filtering is not None, a text summarization technique is
        used for the filtering of <Thread> segments.
    """
    with open(output_fname, "wt") as output_file:
        for orgquestion, (thread, _) \
            in zip(segment_orgquestions(dataset_fnames),
                   segment_threads(dataset_fnames, segment_filtering=segment_filtering)):
            test_score = language_model.similarity(orgquestion, thread)
            test_class = classifier.predict([[test_score]])[0]
            output_file.write("%s\t%s\t0\t%s\t%s\n" % (orgquestion.id, thread.id, repr(test_score),
                                                       "true" if test_class else "false"))

def train_segmented_aggregation(language_model, dataset_fnames, aggregate_tier1_segments,
                                aggregate_tier2_segments, thread_first=True,
                                segment_filtering=None):
    """
        Trains a classifier that maps document similarity to relevance labels.
        The segmented non-ML version computes similarity between segments and
        then performs a reduction step to derive document similarity.

        If full_threads is True, processes entire <Thread>s, otherwise
        processes only the <RelQuestion>s.

        If thread_first is True, the reduction is first performed over <Thread>
        segments and then over <OrgQuestion> segments rather than the other way
        around.

        If segment_filtering is not None, a text summarization technique is
        used for the filtering of <Thread> segments.
    """
    training_scores = []
    training_classes = []
    for orgquestion, (thread, relevant) \
        in zip(segment_orgquestions(dataset_fnames),
               segment_threads(dataset_fnames, segment_filtering=segment_filtering)):
        results = []
        tier1 = thread if thread_first else orgquestion
        tier2 = orgquestion if thread_first else thread
        for tier2_segment in tier2.segments:
            subresults = []
            for tier1_segment in tier1.segments:
                orgquestion_segment = tier2_segment if thread_first else tier1_segment
                thread_segment = tier1_segment if thread_first else tier2_segment
                subresults.append([language_model.similarity(orgquestion_segment, thread_segment),
                                   tier2_segment, tier1_segment])
            subresults_aggregate = aggregate_tier1_segments(subresults, language_model)
            LOGGER.debug("Aggregating subresults: %s -> %s", subresults, subresults_aggregate)
            results.append(subresults_aggregate)
        results_aggregate = aggregate_tier2_segments(results, language_model)
        LOGGER.debug("Aggregating results: %s -> %s", results, results_aggregate)
        training_scores.append(results_aggregate)
        training_classes.append(relevant)
    classifier = LogisticRegression(random_state=LOGISTIC_REGRESSION_RANDOM_STATE)
    classifier.fit(training_scores, training_classes)
    return classifier

def evaluate_segmented_aggregation(language_model, classifier, dataset_fnames, output_fname,
                                   aggregate_tier1_segments, aggregate_tier2_segments,
                                   thread_first=True, segment_filtering=None):
    """
        Produces an output file that contains the ranking of document pairs and
        predicted relevance labels.  The segmented non-ML version computes
        similarity between segments and then performs a reduction step to
        derive document similarity.

        If full_threads is True, processes entire <Thread>s, otherwise
        processes only the <RelQuestion>s.

        If thread_first is True, the reduction is first performed over <Thread>
        segments and then over <OrgQuestion> segments rather than the other way
        around.

        If segment_filtering is not None, a text summarization technique is
        used for the filtering of <Thread> segments.
    """
    with open(output_fname, "wt") as output_file:
        for orgquestion, (thread, _) \
            in zip(segment_orgquestions(dataset_fnames),
                   segment_threads(dataset_fnames, segment_filtering=segment_filtering)):
            results = []
            tier1 = thread if thread_first else orgquestion
            tier2 = orgquestion if thread_first else thread
            for tier2_segment in tier2.segments:
                subresults = []
                for tier1_segment in tier1.segments:
                    orgquestion_segment = tier2_segment if thread_first else tier1_segment
                    thread_segment = tier1_segment if thread_first else tier2_segment
                    subresults.append([language_model.similarity(orgquestion_segment, thread_segment),
                                       tier2_segment, tier1_segment])
                subresults_aggregate = aggregate_tier1_segments(subresults, language_model)
                LOGGER.debug("Aggregating subresults: %s -> %s", subresults, subresults_aggregate)
                results.append(subresults_aggregate)
            results_aggregate = aggregate_tier2_segments(results, language_model)
            LOGGER.debug("Aggregating results: %s -> %s", results, results_aggregate)
            test_score = results_aggregate[0]
            test_class = classifier.predict([[test_score]])[0]
            output_file.write("%s\t%s\t0\t%s\t%s\n" % (orgquestion.id, thread.id, repr(test_score),
                                                       "true" if test_class else "false"))

def train_segmented_ml(language_model, dataset_fnames, segment_filtering=None):
    """
        Trains a classifier that maps document similarity to relevance labels.
        This is done by computing similarity between segments and then
        learning a to classify the segment similarities as relevant / non-relevant.

        If segment_filtering is not None, a text summarization technique is
        used for the filtering of <Thread> segments. Note that the ml approach
        expects all training samples to have the same number of active segments.
    """
    training_scores = []
    training_classes = []
    for orgquestion, (thread, relevant) \
        in zip(segment_orgquestions(dataset_fnames),
               segment_threads(dataset_fnames, segment_filtering=segment_filtering)):
        results = []
        for orgquestion_segment in orgquestion.segments:
            if not orgquestion_segment.active:
                continue
            for thread_segment in thread.segments:
                if not thread_segment.active:
                    continue
                results.append(language_model.similarity(orgquestion_segment, thread_segment))
        training_scores.append(results)
        training_classes.append(relevant)
    classifier = LogisticRegression(random_state=LOGISTIC_REGRESSION_RANDOM_STATE)
    classifier.fit(training_scores, training_classes)
    return classifier

def evaluate_segmented_ml(language_model, classifier, dataset_fnames, output_fname,
                          segment_filtering=None):
    """
        Produces an output file that contains the ranking of document pairs and
        predicted relevance labels.  This is done by computing similarity
        between segments, approximating the relevance of a document pair using
        a pre-learned classifier, and producing a ranking based on the
        classifier certainty.

        If segment_filtering is not None, a text summarization technique is
        used for the filtering of <Thread> segments. Note that the ml approach
        expects all training samples to have the same number of active segments.
    """
    with open(output_fname, "wt") as output_file:
        for orgquestion, (thread, _) \
            in zip(segment_orgquestions(dataset_fnames),
                   segment_threads(dataset_fnames, segment_filtering=segment_filtering)):
            results = []
            for orgquestion_segment in orgquestion.segments:
                if not orgquestion_segment.active:
                    continue
                for thread_segment in thread.segments:
                    if not thread_segment.active:
                        continue
                    results.append(language_model.similarity(orgquestion_segment, thread_segment))
            test_score = classifier.decision_function([results])[0]
            test_class = classifier.predict([results])[0]
            output_file.write("%s\t%s\t0\t%s\t%s\n" % (orgquestion.id, thread.id, repr(test_score),
                                                       "true" if test_class else "false"))
