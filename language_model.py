"""This module contains the language model that maps token lists to vector-space represenations."""

from itertools import chain
import logging
from pickle import load, dump
import re

from gensim import corpora
from numpy import mean

from filenames import UNANNOTATED_DATASET_FNAME, \
    UNANNOTATED_DATASET_DICTIONARY_FNAME as DICTIONARY_FNAME, \
    UNANNOTATED_DATASET_LOG_FNAME as LOG_FNAME, \
    UNANNOTATED_DATASET_PIVOT_STATS_FNAME as PIVOT_STATS_FNAME, \
    UNANNOTATED_DATASET_BM25_STATS_FNAME as BM25_STATS_FNAME, \
    TFIDF_DF_WEIGHTING_METHOD_MAP as DF_WEIGHTING_METHOD_MAP, \
    TFIDF_TF_WEIGHTING_METHOD_MAP as TF_WEIGHTING_METHOD_MAP, \
    TFIDF_NORMALIZATION_METHOD_MAP as NORMALIZATION_METHOD_MAP
from preprocessing import Document, segment_threads
from scoring import bm25, norm_u, norm_b

LOGGER = logging.getLogger(__name__)

class LanguageModel(object):
    """A language model that maps token lists to vector-space represenations."""
    def __init__(self, base_term_weighting="tfidf_ntc_ntc", extra_term_weighting=None):
        """
            Sets up a tf-idf language model using the unannotated SemEval 2016/2017 Task 3 dataset.

            base_term_weighting is either "tfidf_xxx.xxx", where "xxx.xxx" stands for the tf-idf
            SMART notation described in (Salton, Gerard. 1971a), which then determines the base term
            weights during the vectorization, or it is "bm25", which specifies that the
            probabilistic Okapi BM25 scoring will be used instead. You can toggle between the two
            via self.use_tfidf.

            extra_term_weighting specifies additional weighting factors, which stack
            multiplicatively on top of the base term weights during token list vectorization.

            If extra_term_weighting is "godwin", the base weight of a term is multiplied by a factor
            inversely proportional to the sum of the positions at which the term appears in the
            document.

            If extra_term_weighting is "murataetal00_A" or "murataetal00_B", the base weight of a
            term t is multiplied by a factor K_location(d, t) described in (Murata et al., 2000)
            with constants taken for system A or B from section 3. The title and body parameters
            then correspond to the title and body token lists.
        """
        file_handler = logging.FileHandler(LOG_FNAME, encoding='utf8')
        logging.getLogger().addHandler(file_handler)

        # Parse the configuration.
        if re.match(r"tfidf_", base_term_weighting):
            self.use_tfidf = True
        else:
            assert re.match(r"bm25", base_term_weighting)
            self.use_tfidf = False
            self.bm25_k1, self.bm25_k3, self.bm25_b = \
                re.match(r"bm25_k1=([0-9](?:\.[0-9]*)?)_k3=([0-9]*(?:\.[0-9]*)?)_b=([0-9](?:\.[0-9]*)?)",
                         base_term_weighting).groups()
            self.bm25_k1 = float(self.bm25_k1)
            self.bm25_k3 = float(self.bm25_k3)
            self.bm25_b = float(self.bm25_b)
        
        if self.use_tfidf:
            assert extra_term_weighting in (None, "godwin", "murataetal00_A", "murataetal00_B")
        else:
            assert extra_term_weighting is None
        self.extra_term_weighting = extra_term_weighting

        if self.use_tfidf:
            self.tfidf_result = {}
            self.tfidf_query = {}
            self.tfidf_result["tf"], self.tfidf_result["df"], self.tfidf_result["norm"], \
                self.tfidf_slope, self.tfidf_query["tf"], self.tfidf_query["df"], \
                self.tfidf_query["norm"] = re.match(r"tfidf_(.)(.)(.)(?:_s=([0-9](?:\.[0-9]*)?))?_(.)(.)(.)",
                                                    base_term_weighting).groups()
            self.tfidf_result["tf"] = TF_WEIGHTING_METHOD_MAP[self.tfidf_result["tf"]]
            self.tfidf_result["df"] = DF_WEIGHTING_METHOD_MAP[self.tfidf_result["df"]]
            self.tfidf_result["norm"] = NORMALIZATION_METHOD_MAP[self.tfidf_result["norm"]]
            if self.tfidf_result["norm"] in (norm_u, norm_b):
                assert self.tfidf_slope is not None
            if self.tfidf_slope is not None:
                self.tfidf_slope = float(self.tfidf_slope)
            self.tfidf_query["tf"] = TF_WEIGHTING_METHOD_MAP[self.tfidf_query["tf"]]
            self.tfidf_query["df"] = DF_WEIGHTING_METHOD_MAP[self.tfidf_query["df"]]
            self.tfidf_query["norm"] = NORMALIZATION_METHOD_MAP[self.tfidf_query["norm"]]
            assert self.tfidf_query["norm"] not in (norm_u, norm_b)

        # Prepare the BM25 scoring model.
        try:
            with open(BM25_STATS_FNAME, "br") as file:
                self.bm25_avdl = load(file)
        except IOError:
            self.bm25_avdl = {}
            LOGGER.info("preparing the bm25 scoring function statistics")

            self.bm25_avdl["documents"] = mean([sum((len(token) for token in document.tokens)) \
                for document, _ in segment_threads([UNANNOTATED_DATASET_FNAME])])
            LOGGER.info("average document length: %f", self.bm25_avdl["documents"])

            self.bm25_avdl["qsubjects"] = mean([sum((len(token) for token in segment.tokens)) \
                for segment in chain.from_iterable(document.segments for document, _ \
                                       in segment_threads([UNANNOTATED_DATASET_FNAME])) \
                if segment == segment.document.qsubject])
            LOGGER.info("average qsubject segment length: %f", self.bm25_avdl["qsubjects"])

            self.bm25_avdl["qbodies"] = mean([sum((len(token) for token in segment.tokens)) \
                for segment in chain.from_iterable(document.segments for document, _ \
                                       in segment_threads([UNANNOTATED_DATASET_FNAME])) \
                if segment == segment.document.qbody])
            LOGGER.info("average qbody segment length: %f", self.bm25_avdl["qbodies"])

            self.bm25_avdl["comments"] = mean([sum((len(token) for token in segment.tokens)) \
                for segment in chain.from_iterable(document.segments for document, _ \
                                       in segment_threads([UNANNOTATED_DATASET_FNAME])) \
                if segment != segment.document.qsubject and segment != segment.document.qbody])
            LOGGER.info("average comment segment length: %f", self.bm25_avdl["comments"])

            with open(BM25_STATS_FNAME, "bw") as file:
                dump(self.bm25_avdl, file)
            LOGGER.info("done preparing the bm25 scoring function statistics")

        # Prepare the pivoted document normalization tf-idf statistics.
        try:
            with open(PIVOT_STATS_FNAME, "rb") as file:
                self.pivot_stats = load(file)
        except IOError:
            self.pivot_stats = {}
            LOGGER.info("preparing the pivoted document normalization tf-idf statistics")

            self.pivot_stats["documents"] = {}
            self.pivot_stats["documents"]["avgb"] = self.bm25_avdl["documents"]
            self.pivot_stats["documents"]["avgu"] = mean([len(document.terms) \
                for document, _ in segment_threads([UNANNOTATED_DATASET_FNAME])])
            LOGGER.info("average document length: %f", self.pivot_stats["documents"]["avgb"])
            LOGGER.info("average document unique terms: %f", self.pivot_stats["documents"]["avgu"])

            self.pivot_stats["qsubjects"] = {}
            self.pivot_stats["qsubjects"]["avgb"] = self.bm25_avdl["qsubjects"]
            self.pivot_stats["qsubjects"]["avgu"] = mean([len(segment.terms) \
                for segment in chain.from_iterable(document.segments for document, _ \
                                       in segment_threads([UNANNOTATED_DATASET_FNAME])) \
                if segment == segment.document.qsubject])
            LOGGER.info("average qsubject segment length: %f",
                        self.pivot_stats["qsubjects"]["avgb"])
            LOGGER.info("average qsubject segment unique terms: %f",
                        self.pivot_stats["qsubjects"]["avgu"])

            self.pivot_stats["qbodies"] = {}
            self.pivot_stats["qbodies"]["avgb"] = self.bm25_avdl["qbodies"]
            self.pivot_stats["qbodies"]["avgu"] = mean([len(segment.terms) \
                for segment in chain.from_iterable(document.segments for document, _ \
                                       in segment_threads([UNANNOTATED_DATASET_FNAME])) \
                if segment == segment.document.qbody])
            LOGGER.info("average qbody segment length: %f", self.pivot_stats["qbodies"]["avgb"])
            LOGGER.info("average qbody segment unique terms: %f",
                        self.pivot_stats["qbodies"]["avgu"])

            self.pivot_stats["comments"] = {}
            self.pivot_stats["comments"]["avgb"] = self.bm25_avdl["comments"]
            self.pivot_stats["comments"]["avgu"] = mean([len(segment.terms) \
                for segment in chain.from_iterable(document.segments for document, _ \
                                       in segment_threads([UNANNOTATED_DATASET_FNAME])) \
                if segment != segment.document.qsubject and segment != segment.document.qbody])
            LOGGER.info("average comment segment length: %f", self.pivot_stats["comments"]["avgb"])
            LOGGER.info("average comment segment unique terms: %f",
                        self.pivot_stats["comments"]["avgu"])

            with open(PIVOT_STATS_FNAME, "wb") as file:
                dump(self.pivot_stats, file)
            LOGGER.info("done preparing the pivoted document normalization tf-idf statistics")

        # Prepare the dictionary.
        try:
            self.dictionary = corpora.Dictionary.load(DICTIONARY_FNAME, mmap='r')
        except IOError:
            self.dictionary = \
                corpora.Dictionary(segment for segment in chain.from_iterable( \
                    document.segments for document, _ \
                                      in segment_threads([UNANNOTATED_DATASET_FNAME])))
            self.dictionary.save(DICTIONARY_FNAME)

        logging.getLogger().removeHandler(file_handler)

    def vectorize(self, segment, is_query=False):
        """
            Returns a vector representation of a segment (or a document) for
            tf-idf similarity scoring.

            is_query determines, whether the tf-idf weighting scheme for
            queries will be used rather than the tf-idf weighting scheme for
            results.
        """
        segment_bow = self.dictionary.doc2bow(segment.tokens)

        # Perform base weighting.
        base_term_weights = {}
        tfs = [term_frequency for _, term_frequency in segment_bow]
        for term_id, tf in segment_bow:
            df = self.dictionary.dfs[term_id]
            N = self.dictionary.num_docs
            model = self.tfidf_query if is_query else self.tfidf_result
            base_term_weights[term_id] = model["tf"](tf, tfs) * model["df"](df, N)

        # Perform extra weighting.
        extra_term_weights = {}
        if self.extra_term_weighting:
            if self.extra_term_weighting == "murataetal00_A":
                k_location_1 = 1.35
                k_location_2 = 0.125
            elif self.extra_term_weighting == "murataetal00_B":
                k_location_1 = 1.3
                k_location_2 = 0.15
            for (token_position, token) in enumerate(segment.tokens):
                token_bow = self.dictionary.doc2bow([token])
                if len(token_bow) == 0:
                    continue # An out-of-dictionary token
                [(token_id, _)] = token_bow
                if self.extra_term_weighting == "godwin":
                    if token_id not in extra_term_weights:
                        extra_term_weights[token_id] = 0.0
                    extra_term_weights[token_id] += len(segment.tokens) / (token_position+1)
                elif self.extra_term_weighting == "murataetal00_A" \
                     or self.extra_term_weighting == "murataetal00_B":
                    if token_id not in extra_term_weights:
                        if segment.document.murataetal00["P"][token] == "title":
                            extra_term_weights[token_id] = k_location_1
                        else:
                            extra_term_weights[token_id] = \
                                1 + k_location_2 * (segment.murataetal00["length_d"] \
                                - 2 * segment.document.murataetal00["P"][token]) \
                                / segment.murataetal00["length_d"]

        return [(term_id, base_term_weights[term_id] * (extra_term_weights[term_id] \
                                                        if extra_term_weights else 1.0)) \
                for term_id, _ in segment_bow]

    def similarity(self, query, result):
        """
            Returns cosine similarity between two document segments (or documents). Note that if
            different tf-idf weighting is used for query and result vectors, or when the
            probabilistic BM25 scoring is used, this function is not symmetric.
        """
        if self.use_tfidf:
            # Compute similarity using the tf-idf framework.
            query_vector = self.vectorize(query, is_query=True)
            result_vector = self.vectorize(result, is_query=False)
            if isinstance(result, Document):
                pivot_stats = self.pivot_stats["documents"]
            else:
                assert result in result.document.segments
                if result.document.qsubject == result:
                    pivot_stats = self.pivot_stats["qsubjects"]
                elif result.document.qbody == result:
                    pivot_stats = self.pivot_stats["qbodies"]
                else:
                    pivot_stats = self.pivot_stats["comments"]
            query_norm = self.tfidf_query["norm"](query_vector, None, None, None)
            result_norm = self.tfidf_result["norm"]( \
                result_vector, {"avgu": pivot_stats["avgu"], "avgb": pivot_stats["avgb"],
                                "u": len(result.terms),
                                "b": sum((len(token) for token in result.tokens))},
                self.tfidf_slope)
            result_term_weights = dict(result_vector)
            numerator = sum((query_term_weight * result_term_weights[term_id] \
                             for term_id, query_term_weight in query_vector \
                             if term_id in result_term_weights))
            return numerator / (query_norm * result_norm) if numerator > 0.0 else 0.0
        else:
            # Compute similarity using the probabilistic BM25 scoring.
            tfs = dict(self.dictionary.doc2bow(result.tokens))
            qtfs = dict(self.dictionary.doc2bow(query.tokens))
            if isinstance(result, Document):
                avdl = self.bm25_avdl["documents"]
            else:
                assert result in result.document.segments
                if result.document.qsubject == result:
                    avdl = self.bm25_avdl["qsubjects"]
                elif result.document.qbody == result:
                    avdl = self.bm25_avdl["qbodies"]
                else:
                    avdl = self.bm25_avdl["comments"]
            dl = sum((len(token) for token in result.tokens))
            return sum((bm25(tfs[term_id], qtf, self.dictionary.num_docs, \
                             self.dictionary.dfs[term_id], dl, avdl, \
                             k1=self.bm25_k1, k3=self.bm25_k3, b=self.bm25_b) \
                        for term_id, qtf in qtfs.items() if term_id in tfs))
