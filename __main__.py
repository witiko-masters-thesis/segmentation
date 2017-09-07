"""This module implements the command-line interface."""

import logging
from sys import argv
import re

from filenames import SUBTASK_B_TRAIN2016_DATASET_FNAMES as TRAIN2016_DATASET_FNAMES, \
    SUBTASK_B_TRAIN2017_DATASET_FNAMES as TRAIN2017_DATASET_FNAMES, TEST2016_DATASET_FNAME, \
    TEST2016_DIRNAME, TEST2017_DATASET_FNAME, TEST2017_DIRNAME, TEST_PREDICTIONS_BASE_DIRNAME, \
    TEST2016_PREDICTIONS_DIRNAME, TEST2017_PREDICTIONS_DIRNAME, TEST2016_GOLD_BASE_FNAME, \
    TEST2017_GOLD_BASE_FNAME, AGGREGATION_METHOD_MAP, DEV_DATASET_FNAME, DEV_GOLD_BASE_FNAME
from evaluation import train_nonsegmented, train_segmented_aggregation, train_segmented_ml, \
    evaluate_nonsegmented, evaluate_segmented_aggregation, evaluate_segmented_ml, \
    produce_gold_results
from language_model import LanguageModel

LOGGER = logging.getLogger(__name__)

def main():
    """This function implements the command-line interface."""
    # Parse input configuration.
    if argv[1] == "prepare":
        logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                            level=logging.INFO)
        # Prepare the language model.
        language_model = LanguageModel()
        # Produce the gold results for the dev dataset.
        produce_gold_results([DEV_DATASET_FNAME],
                             "%s/%s" % (TEST2016_DIRNAME, DEV_GOLD_BASE_FNAME))
        raise SystemExit
    else:
        logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                            level=logging.WARNING)
    config = argv[1].split('-')
    method = config[0]
    assert method in ("unsegmented", "segmented_ml", "segmented_aggregation")
    year = argv[2]
    assert year in ("dev", "2016", "2017")
    segment_filtering_method = config[1]
    assert segment_filtering_method in \
        ("none", "kolczetal00_title", "kolczetal00_firstpara",
         "kolczetal00_parawithmosttitlewords", "kolczetal00_firsttwopara",
         "kolczetal00_firstlastpara") \
        or re.match(r"kolczetal00_bestsentence[0-5]", segment_filtering_method)
    segment_filtering = segment_filtering_method \
                        if segment_filtering_method != "none" else None
    base_term_weighting = config[2]
    assert re.match(r"(bm25|tfidf)_", base_term_weighting)
    extra_term_weighting_method = config[3]
    assert extra_term_weighting_method in ("none", "godwin", "murataetal00_A",
                                           "murataetal00_B")
    extra_term_weighting = extra_term_weighting_method \
                           if extra_term_weighting_method != "none" else None
    if method == "segmented_aggregation":
        aggregate_tier1_segments_method = config[4]
        assert aggregate_tier1_segments_method in AGGREGATION_METHOD_MAP.keys()
        aggregate_tier1_segments = AGGREGATION_METHOD_MAP[aggregate_tier1_segments_method]
        aggregate_tier2_segments_method = config[5]
        assert aggregate_tier2_segments_method in AGGREGATION_METHOD_MAP.keys()
        aggregate_tier2_segments = AGGREGATION_METHOD_MAP[aggregate_tier2_segments_method]
        order = config[6]
        assert order in ("result_first", "query_first")
        thread_first = order == "result_first"

    # Determine directory and file names
    if year == "dev":
        test_dirname = TEST2016_DIRNAME
        test_predictions_dirname = TEST2016_PREDICTIONS_DIRNAME
        gold_base_fname = DEV_GOLD_BASE_FNAME
        test_dataset_fname = DEV_DATASET_FNAME
        train_dataset_fnames = TRAIN2016_DATASET_FNAMES
    elif year == "2016":
        test_dirname = TEST2016_DIRNAME
        test_predictions_dirname = TEST2016_PREDICTIONS_DIRNAME
        gold_base_fname = TEST2016_GOLD_BASE_FNAME
        test_dataset_fname = TEST2016_DATASET_FNAME
        train_dataset_fnames = TRAIN2016_DATASET_FNAMES + [DEV_DATASET_FNAME]
    elif year == "2017":
        test_dirname = TEST2017_DIRNAME
        test_predictions_dirname = TEST2017_PREDICTIONS_DIRNAME
        gold_base_fname = TEST2017_GOLD_BASE_FNAME
        test_dataset_fname = TEST2017_DATASET_FNAME
        train_dataset_fnames = TRAIN2017_DATASET_FNAMES + [DEV_DATASET_FNAME]
    output_fname = "%s/subtask_B_%s-%s.txt" % (test_predictions_dirname, argv[1], argv[2])
    base_output_fname = "%s/subtask_B_%s-%s.txt" % (TEST_PREDICTIONS_BASE_DIRNAME, argv[1], argv[2])
    LOGGER.info("Producing %s ...", output_fname)

    # Perform training
    language_model = LanguageModel(base_term_weighting=base_term_weighting,
                                   extra_term_weighting=extra_term_weighting)
    if method == "segmented_ml":
        classifier = train_segmented_ml(language_model, train_dataset_fnames,
                                        segment_filtering=segment_filtering)
    elif method == "segmented_aggregation":
        classifier = train_segmented_aggregation(language_model, train_dataset_fnames,
                                                 aggregate_tier1_segments,
                                                 aggregate_tier2_segments,
                                                 thread_first=thread_first,
                                                 segment_filtering=segment_filtering)
    elif method == "unsegmented":
        classifier = train_nonsegmented(language_model, train_dataset_fnames,
                                        segment_filtering=segment_filtering)

    # Perform evaluation
    if method == "segmented_ml":
        evaluate_segmented_ml(language_model, classifier, [test_dataset_fname], output_fname,
                              segment_filtering=segment_filtering)
    elif method == "segmented_aggregation":
        evaluate_segmented_aggregation(language_model, classifier,
                                       [test_dataset_fname], output_fname,
                                       aggregate_tier1_segments,
                                       aggregate_tier2_segments,
                                       thread_first=thread_first,
                                       segment_filtering=segment_filtering)
    elif method == "unsegmented":
        evaluate_nonsegmented(language_model, classifier, [test_dataset_fname], output_fname,
                              segment_filtering=segment_filtering)

    print("%s %s %s" % (test_dirname, gold_base_fname, base_output_fname))

if __name__ == "__main__":
    main()
