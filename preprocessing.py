"""This module provides functions for parsing SemEval 2016/2017 Task 3 datasets."""

from itertools import chain
import logging
import re
import xml.etree.ElementTree as ElementTree

from gensim.utils import simple_preprocess

CLEANUP_REGEXES = {
    'html': r'<[^<>]+(>|$)',
    'tags': r'\[img_assist[^]]*?\]',
    'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
}

LOGGER = logging.getLogger(__name__)

class Document(object):
    """
        A document object that corresponds to <Thread> or <OrgQuestion>
        elements from SemEval 2016/2017 Task 3 datasets.
    """
    def __init__(self, id, segments, qbody, qsubject, segment_filtering=None):
        """
            Sets up a document object that corresponds to <Thread> or
            <OrgQuestion> elements from SemEval 2016/2017 Task 3 datasets.

            id is the unique identifier of a document.

            segments is a list of text segments in the document. Under our
            model, segments correspond to the <OrgQSubject>, <OrgQBody>,
            <RelQSubject>, <RelQBody>, and <RelCText> XML elements.

            qbody is a text segment corresponding to either the
            <OrgQBody>, or the <RelQBody> XML element.

            qsubject is a text segment corresponding to either the
            <OrgQSubject>, or the <RelQsubjec> XML element.

            segment_filtering specifies which method will be used to filter
            out non-salient document segments.

            self.mutataetal00 contains statistics that are used by the language
            model for the term weighting based on the (Murata et al., 2000)
            article.

            self.terms contains a set of terms that appear in the segment and
            self.tokens contains a list of tokens that appear in the segment.

            self.document refers back to self. This allows Document object
            to act as Segment objects in certain situations, such as similarity
            computations.
        """
        assert isinstance(id, str) and isinstance(segments, list) \
               and isinstance(qsubject, Segment) and isinstance(qbody, Segment)
        for segment in segments:
            assert segment.document is None
        self.id = id
        self.segments = segments
        for segment in self.segments:
            segment.document = self
        self.qsubject = qsubject
        self.qbody = qbody
        self.document = self

        # Pre-compute statistics for murataelal00 term weighting.
        term_positions = {}
        for term in qsubject.terms:
            if term not in term_positions:
                term_positions[term] = "title"
        for token_position, token in enumerate(chain(*(segment.tokens for segment in segments \
                                                       if segment != qsubject))):
            if token not in term_positions:
                term_positions[token] = token_position
        self.murataetal00 = {
            "P": term_positions,
            "length_d": sum(len(segment.tokens) for segment in segments if segment != qsubject)
        }

        # Perform segment filtering.
        assert segment_filtering in [None, "kolczetal00_title", "kolczetal00_firstpara",
                                     "kolczetal00_parawithmosttitlewords",
                                     "kolczetal00_firsttwopara",
                                     "kolczetal00_firstlastpara"] \
               or re.match(r"kolczetal00_bestsentence[0-5]", segment_filtering)
        if segment_filtering != None:
            if segment_filtering == "kolczetal00_title":
                for segment in segments:
                    segment.active = segment == qsubject
            elif segment_filtering == "kolczetal00_firstpara":
                for segment in segments:
                    segment.active = segment == qsubject or segment == qbody
            elif segment_filtering == "kolczetal00_parawithmosttitlewords":
                best_segment = sorted([segment for segment in segments if segment != qsubject],
                                      key=lambda segment: len([token for token in segment.tokens \
                                                               if token in qsubject.terms]),
                                      reverse=True)[0]
                for segment in segments:
                    segment.active = segment == qsubject or segment == best_segment
            elif segment_filtering == "kolczetal00_firsttwopara":
                for segment in segments:
                    segment.active = segment == qsubject or segment == qbody \
                                     or segment == segments[2]
            elif segment_filtering == "kolczetal00_firstlastpara":
                for segment in segments:
                    segment.active = segment == qsubject or segment == qbody \
                                     or segment == segments[-1]
            elif re.match(r"kolczetal00_bestsentence[0-5]", segment_filtering):
                min_common_tokens = int(re.match(r"kolczetal00_bestsentence([0-5])",
                                                 segment_filtering).group(1)) 
                best_segments = (segment for segment in segments \
                                 if segment != qsubject \
                                    and len([token for token in segment.tokens \
                                             if token in qsubject.terms]) > min_common_tokens)
                for segment in segments:
                    segment.active = segment == qsubject
                for segment in best_segments:
                    segment.active = True

        # Extract terms and tokens from active segments.
        self.tokens = []
        self.terms = set()
        for segment in segments:
            if segment.active:
                self.tokens.extend(segment.tokens)
                self.terms.update(segment.terms)

    def __str__(self):
        return ' '.join(self.tokens).__str__()

    def __repr__(self):
        return ' '.join(self.tokens).__repr__()

class Segment(object):
    """
        A document segment object that corresponds to the
        <OrgQSubject>, <OrgQBody>, <RelQSubject>, <RelQBody>, or <RelCText>
        XML element from SemEval 2016/2017 Task 3 datasets.
    """
    def __init__(self, text):
        """
            Sets up a document segment object that corresponds to the
            <OrgQSubject>, <OrgQBody>, <RelQSubject>, <RelQBody>, or <RelCText>
            XML element from SemEval 2016/2017 Task 3 datasets.

            text is the raw unaltered text content of the XML element, which
            is cleaned up and transformed to a list of tokens self.tokens.

            Each segment can be either active, or filtered out, as indicated
            by the boolean value of self.active. Each segment also belongs to
            at most one document indicated by self.document.

            self.mutataetal00 contains statistics that are used by the language
            model for the term weighting based on the (Murata et al., 2000)
            article.

            self.terms contains a set of terms that appear in the segment and
            self.tokens contains a list of tokens that appear in the segment.
        """
        assert text is None or isinstance(text, str)
        if text is None:
            self.tokens = []
        else:
            for pattern in CLEANUP_REGEXES.values():
                text = re.sub(pattern, '', text)
            self.tokens = simple_preprocess(text)
        self.terms = set(self.tokens)
        self.active = True
        self.document = None

        # Pre-compute statistics for murataelal00 term weighting.
        self.murataetal00 = {
            "length_d": len(self.tokens)
        }

    def __str__(self):
        return ' '.join(self.tokens).__str__()

    def __repr__(self):
        return ' '.join(self.tokens).__repr__()

def segment_orgquestions(dataset_fnames):
    """Segments <OrgQuestion> elements from SemEval 2016/2017 Task 3 datasets."""
    qbody = None
    qsubject = None
    for dataset_fname in dataset_fnames:
        for event, elem in ElementTree.iterparse(dataset_fname):
            if event == "end":
                if elem.tag == "OrgQSubject" or elem.tag == "OrgQBody":
                    segment = Segment(elem.text)
                    if elem.tag == "OrgQSubject":
                        qsubject = segment
                    else:
                        qbody = segment
                elif elem.tag == "OrgQuestion":
                    id = elem.attrib["ORGQ_ID"]
                    assert qbody is not None and qsubject is not None
                    yield Document(id, [qbody, qsubject], qbody, qsubject)
                    qbody = None
                    qsubject = None
            elem.clear()

def segment_threads(dataset_fnames, segment_filtering=None):
    """
        Segments <Thread> elements from SemEval 2016/2017 Task 3 datasets into token lists.
        If full_threads=True, processes entire <Thread>s, otherwise processes
        only the <RelQuestion>s.

        If segment_filtering is not None, a text summarization technique is
        used for the filtering of <Thread> segments.
    """
    segments = []
    relevant = None
    qbody = None
    qsubject = None
    for dataset_fname in dataset_fnames:
        for event, elem in ElementTree.iterparse(dataset_fname):
            if event == "end":
                if elem.tag == "RelQSubject" or elem.tag == "RelQBody" or elem.tag == "RelCText":
                    segment = Segment(elem.text)
                    if elem.tag == "RelQSubject":
                        qsubject = segment
                    if elem.tag == "RelQBody":
                        qbody = segment
                    elif elem.tag == "RelCText":
                        assert segments
                        segments.append(segment)
                elif elem.tag == "RelQuestion":
                    if "RELQ_RELEVANCE2ORGQ" in elem.attrib:
                        relevance_label = elem.attrib["RELQ_RELEVANCE2ORGQ"]
                        relevant = relevance_label == "PerfectMatch" \
                                   or relevance_label == "Relevant"
                    assert qbody is not None and qsubject is not None
                    segments.append(qbody)
                    segments.append(qsubject)
                elif elem.tag == "Thread":
                    id = elem.attrib["THREAD_SEQUENCE"]
                    yield (Document(id, segments, qbody, qsubject, \
                                    segment_filtering=segment_filtering),
                           relevant)
                    segments = []
                    qbody = None
                    qsubject = None
            elem.clear()

def retrieve_comment_relevancies(dataset_fnames):
    """
        Extracts the RELC_RELEVANCE2RELQ attributes in <Thread> elements from
        SemEval 2016/2017 Task 3 subtask A datasets.
    """
    relevancies = []
    for dataset_fname in dataset_fnames:
        for event, elem in ElementTree.iterparse(dataset_fname):
            if event == "end":
                if elem.tag == "RelComment":
                    relevance_label = elem.attrib["RELC_RELEVANCE2RELQ"]
                    relevant = relevance_label == "Good" \
                            or relevance_label == "PotentiallyUseful"
                    relevancies.append(relevant)
                elif elem.tag == "Thread":
                    yield relevancies
                    relevancies = []
            elem.clear()
