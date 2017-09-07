#!/bin/bash
set -e
export LC_ALL=C
export PARALLEL_SHELL=/bin/bash
export PYTHONHASHSEED=12345

cd ..
(for K1 in `seq 1.0 0.05 2.0`; do
  for K3 in `seq 0 50 1000`; do
    for B in `seq 0.0 0.05 1.0`; do
      # Okapi BM25 grid search
      BASE_TERM_WEIGHTING="bm25_k1=${K1}_k3=${K3}_b=$B"
      printf 'segmented_aggregation-none-%s-none-max-wavg_koetal04-query_first\n' "$BASE_TERM_WEIGHTING"
      printf 'segmented_ml-kolczetal00_firsttwopara-%s-none\n' "$BASE_TERM_WEIGHTING"
      printf 'unsegmented-kolczetal00_title-%s-none\n' "$BASE_TERM_WEIGHTING"
    done
  done
done
for S in `seq 0.0 0.05 1.0`; do
  # Tf-idf pivoted document length normalization grid search
  printf 'segmented_aggregation-kolczetal00_firsttwopara-tfidf_Lpb_s=%s_bfc-murataetal00_B-avg-wavg_length-result_first\n' "$S"
  printf 'segmented_aggregation-kolczetal00_firsttwopara-tfidf_Lpu_s=%s_Lpc-murataetal00_B-avg-wavg_length-result_first\n' "$S"
  printf 'segmented_ml-kolczetal00_firsttwopara-tfidf_Lfb_s=%s_bfc-murataetal00_B\n' "$S"
  printf 'segmented_ml-kolczetal00_firsttwopara-tfidf_Lpu_s=%s_Lpc-murataetal00_B\n' "$S"
  printf 'unsegmented-none-tfidf_Lpu_s=%s_Lpc-murataetal00_A\n' "$S"
  printf 'unsegmented-none-tfidf_dnb_s=%s_dtn-murataetal00_B\n' "$S"
done) | parallel --halt=2 --bar -- '
  set -e
  RESULTS="$(python3 __main__.py {} dev)"
  read TEST_DIRNAME GOLD_BASE_FNAME BASE_OUTPUT_FNAME < <(echo $RESULTS)
  cd $TEST_DIRNAME
  python2 _scorer/ev.py $GOLD_BASE_FNAME $BASE_OUTPUT_FNAME | tee $BASE_OUTPUT_FNAME.score \
    | sed -n -r "/^ALL SCORES:/{s/^ALL SCORES:/{}/;s/\t/,/g;s/^([^,]*(,[^,]*){3,3}),.*/\1/;p}"
' | tee "$OLDPWD"/results-unsorted.csv | sort -r -t, -k 2 >"$OLDPWD"/results.csv
