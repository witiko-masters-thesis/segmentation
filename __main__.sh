#!/bin/bash
set -e
export LC_ALL=C
export PARALLEL_SHELL=/bin/bash

python3 __main__.py prepare
for YEAR in dev 2016 2017; do 
  echo config,MAP,AvgRec,MRR
  # Print the SemEval-Task3 baselines.
  if [[ $YEAR = 2016 ]]; then
    echo baseline_1_IR,0.7475,0.8830,83.79
    echo baseline_2_random,0.4698,0.6792,50.96
  elif [[ $YEAR = 2017 ]]; then
    echo baseline_1_IR,0.4185,0.7759,46.42
    echo baseline_2_random,0.2981,0.6265,33.02
  fi
  for METHOD in {unsegmented,segmented_{ml,aggregation}}; do
    # Set up optimal pivoted document normalization slopes.
    case $METHOD in
      segmented_aggregation)
        U_S=0.15;
        B_S=0.20;;
      segmented_ml)
        U_S=0.25;
        B_S=0.35;;
      unsegmented)
        U_S=0.30
        B_S=0.30;;
      *)
        exit 1;;
    esac
    for SEGMENT_FILTERING in {none,kolczetal00_{title,firstpara,parawithmosttitlewords,firsttwopara,firstlastpara,bestsentence{0..5}}}; do
      if [[ $METHOD = segmented_ml && $SEGMENT_FILTERING =~ ^kolczetal00_bestsentence ]]; then
        continue # kolczetal00_bestsentence produces a variable number of segments, which segmented_ml cannot handle.
      fi
      for TERM_WEIGHTING in tfidf_{nfc_nfc,nfx_nfx,bfx_nfx,lnc_ltc,dnb_s=${B_S}_dtn,dtb_s=${B_S}_nnn,nfc_lfc,nfc_dfc,Lpc_anc,Lpc_ann,dpc_ann,Lpu_s=${U_S}_Lpc,Lfb_s=${B_S}_bfc,Lpb_s=${B_S}_bfc,nnc_bfc,npc_bpn,npc_bfn}-{none,godwin,murataetal00_{A,B}} bm25_{k1=1.2_k3=1000.0_b=0.75,k1=1.2_k3=0_b=0.75,k1=2.00_k3=950_b=0,k1=2.00_k3=0_b=0.80}-none; do
        if [[ $METHOD = segmented_aggregation ]]; then
          # Evaluate aggregation operators invariant to aggregation order.
          for AGGREGATION_TIER1_OPERATOR in {avg,wavg_{length,godwin,koetal04}}; do
            for AGGREGATION_TIER2_OPERATOR in {avg,wavg_{length,godwin,koetal04}}; do
              # Linear aggregation operators are invariant to the aggregation order.
              printf '%s-%s-%s-%s-%s-result_first\n' $METHOD $SEGMENT_FILTERING \
                $TERM_WEIGHTING $AGGREGATION_TIER1_OPERATOR $AGGREGATION_TIER2_OPERATOR
            done
          done
          printf '%s-%s-%s-min-min-result_first\n' $METHOD $SEGMENT_FILTERING $TERM_WEIGHTING
          printf '%s-%s-%s-max-max-result_first\n' $METHOD $SEGMENT_FILTERING $TERM_WEIGHTING
          # Evaluate aggregation operators variant to aggregation order.
          for AGGREGATION_ORDER in {result,query}_first; do
            for AGGREGATION_TIER1_OPERATOR in {min,max}; do
              for AGGREGATION_TIER2_OPERATOR in {avg,wavg_{length,godwin,koetal04}}; do
                printf '%s-%s-%s-%s-%s-%s\n' $METHOD $SEGMENT_FILTERING $TERM_WEIGHTING \
                  $AGGREGATION_TIER1_OPERATOR $AGGREGATION_TIER2_OPERATOR $AGGREGATION_ORDER
              done
            done
            for AGGREGATION_TIER1_OPERATOR in {avg,wavg_{length,godwin,koetal04}}; do
              for AGGREGATION_TIER2_OPERATOR in {min,max}; do
                printf '%s-%s-%s-%s-%s-%s\n' $METHOD $SEGMENT_FILTERING $TERM_WEIGHTING \
                  $AGGREGATION_TIER1_OPERATOR $AGGREGATION_TIER2_OPERATOR $AGGREGATION_ORDER
              done
            done
          done
        else
          printf '%s-%s-%s\n' $METHOD $SEGMENT_FILTERING $TERM_WEIGHTING
        fi
      done
    done
  done | parallel --halt=2 --bar -- '
    set -e
    RESULTS="$(python3 __main__.py {} '$YEAR')"
    read TEST_DIRNAME GOLD_BASE_FNAME BASE_OUTPUT_FNAME < <(echo $RESULTS)
    cd $TEST_DIRNAME
    python2 _scorer/ev.py $GOLD_BASE_FNAME $BASE_OUTPUT_FNAME | tee $BASE_OUTPUT_FNAME.score \
      | sed -n -r "/^ALL SCORES:/{s/^ALL SCORES:/{}/;s/\t/,/g;s/^([^,]*(,[^,]*){3,3}),.*/\1/;p}"
  ' | tee results-${YEAR}_unsorted.csv | sort -r -t, -k 2 >results-${YEAR}.csv
done
