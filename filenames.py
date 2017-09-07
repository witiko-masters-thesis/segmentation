"""This module records filenames and mappings from config strings to functions."""

from aggregation import aggregate_min, aggregate_max, aggregate_avg, aggregate_wavg_length, \
    aggregate_wavg_godwin, aggregate_wavg_koetal04
from scoring import df_n, df_t, df_f, df_F, df_p, tf_n, tf_l, tf_a, tf_b, tf_L, tf_d, norm_n, \
    norm_c, norm_u, norm_b

# The following constant contain filenames and pathnames related to datasets.
TRAIN_DATASET_DIRNAME = "datasets/v3.2/train"
SUBTASK_B_TRAIN2016_DATASET_FNAMES = \
    ["%s/SemEval2016-Task3-CQA-QL-train-part1.xml" % TRAIN_DATASET_DIRNAME,
     "%s/SemEval2016-Task3-CQA-QL-train-part2.xml" % TRAIN_DATASET_DIRNAME]
DEV_DATASET_DIRNAME = "datasets/v3.2/dev"
DEV_DATASET_FNAME = "%s/SemEval2016-Task3-CQA-QL-dev.xml" % DEV_DATASET_DIRNAME
TEST2016_DATASET_DIRNAME = "datasets/SemEval2016_task3_test/English"
TEST2016_DATASET_FNAME = "%s/SemEval2016-Task3-CQA-QL-test.xml" % TEST2016_DATASET_DIRNAME
SUBTASK_B_TRAIN2017_DATASET_FNAMES = SUBTASK_B_TRAIN2016_DATASET_FNAMES + [TEST2016_DATASET_FNAME]
TEST2016_DIRNAME = "datasets/SemEval2016_task3_submissions_and_scores"
TEST2017_DATASET_DIRNAME = "datasets/SemEval2017_task3_test_input_ABCD/English-ABC"
TEST2017_DATASET_FNAME = "%s/SemEval2017-task3-English-test-input.xml" % TEST2017_DATASET_DIRNAME
TEST2017_DIRNAME = "datasets/SemEval2017_task3_submissions_and_scores"
SUBTASK_A_TRAIN_DATASET_FNAMES = \
    ["%s/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml" % TRAIN_DATASET_DIRNAME,
     "%s/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml" % TRAIN_DATASET_DIRNAME,
     "%s/SemEval2016-Task3-CQA-QL-test-subtaskA.xml" % TEST2016_DATASET_DIRNAME,
     "%s/SemEval2017-task3-English-test-subtaskA-input.xml" % TEST2017_DATASET_DIRNAME]
TEST_PREDICTIONS_BASE_DIRNAME = "RaRe"
TEST2016_PREDICTIONS_DIRNAME = "%s/%s" % (TEST2016_DIRNAME, TEST_PREDICTIONS_BASE_DIRNAME)
TEST2017_PREDICTIONS_DIRNAME = "%s/%s" % (TEST2017_DIRNAME, TEST_PREDICTIONS_BASE_DIRNAME)
DEV_GOLD_BASE_FNAME = "../SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy"
TEST2016_GOLD_BASE_FNAME = "_gold/SemEval2016-Task3-CQA-QL-test.xml.subtaskB.relevancy"
TEST2017_GOLD_BASE_FNAME = "_gold/SemEval2017-Task3-CQA-QL-test.xml.subtaskB.relevancy"
UNANNOTATED_DATASET_BASE_FNAME = "datasets/QL-unannotated-data-subtaskA"
UNANNOTATED_DATASET_FNAME = "%s.xml" % UNANNOTATED_DATASET_BASE_FNAME
UNANNOTATED_DATASET_DICTIONARY_FNAME = "%s.dict" % UNANNOTATED_DATASET_BASE_FNAME
UNANNOTATED_DATASET_PIVOT_STATS_FNAME = "%s.pivot" % UNANNOTATED_DATASET_BASE_FNAME
UNANNOTATED_DATASET_BM25_STATS_FNAME = "%s.bm25" % UNANNOTATED_DATASET_BASE_FNAME
UNANNOTATED_DATASET_LOG_FNAME = "%s.log" % UNANNOTATED_DATASET_BASE_FNAME

# The following constants contain mapping from configuration strings to functions.
AGGREGATION_METHOD_MAP = \
    {"min": aggregate_min, "max": aggregate_max, "avg": aggregate_avg,
     "wavg_length": aggregate_wavg_length, "wavg_godwin": aggregate_wavg_godwin,
     "wavg_koetal04": aggregate_wavg_koetal04}
TFIDF_TF_WEIGHTING_METHOD_MAP = \
    {'n': tf_n, 'l': tf_l, 'a': tf_a, 'b': tf_b, 'L': tf_L, 't': tf_n, 'd': tf_d}
TFIDF_DF_WEIGHTING_METHOD_MAP = {'n': df_n, 'x': df_n, 'f': df_f, 't': df_t, 'F': df_F, 'p': df_p}
TFIDF_NORMALIZATION_METHOD_MAP = {'n': norm_n, 'x': norm_n, 'c': norm_c, 'u': norm_u, 'b': norm_b}
