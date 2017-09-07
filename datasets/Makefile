.PHONY: all
all: QL-unannotated-data-subtaskA.xml v3.2/dev/SemEval2016-Task3-CQA-QL-dev.xml \
	SemEval2016_task3_test/English/SemEval2016-Task3-CQA-QL-test.xml \
	SemEval2016_task3_submissions_and_scores/_scorer/ev.py \
	SemEval2017_task3_test_input_ABCD/English-ABC/SemEval2017-task3-English-test-input.xml \
	SemEval2017_task3_submissions_and_scores/_scorer/ev.py

QL-unannotated-data-subtaskA.xml.zip semeval2016-task3-cqa-ql-traindev-v3.2.zip semeval2016_task3_submissions_and_scores.zip semeval2016_task3_test.zip:
	wget http://alt.qcri.org/semeval2016/task3/data/uploads/$@

semeval2017_task3_test_input_abcd.zip semeval2017_task3_submissions_and_scores.zip:
	wget http://alt.qcri.org/semeval2017/task3/data/uploads/$@
	
QL-unannotated-data-subtaskA.xml: QL-unannotated-data-subtaskA.xml.zip
	unzip -o $<
	test -e $@ && touch $@

v3.2/dev/SemEval2016-Task3-CQA-QL-dev.xml: semeval2016-task3-cqa-ql-traindev-v3.2.zip
	unzip -o $<
	test -e $@ && touch $@

SemEval2016_task3_test/English/SemEval2016-Task3-CQA-QL-test.xml: semeval2016_task3_test.zip
	unzip -o $<
	test -e $@ && touch $@

SemEval2016_task3_submissions_and_scores/_scorer/ev.py: semeval2016_task3_submissions_and_scores.zip
	unzip -o $<
	mkdir SemEval2016_task3_submissions_and_scores/RaRe
	test -e $@ && touch $@

SemEval2017_task3_test_input_ABCD/English-ABC/SemEval2017-task3-English-test-input.xml: semeval2017_task3_test_input_abcd.zip
	unzip -o $<
	test -e $@ && touch $@

SemEval2017_task3_submissions_and_scores/_scorer/ev.py: semeval2017_task3_submissions_and_scores.zip
	unzip -o $<
	mkdir SemEval2017_task3_submissions_and_scores/RaRe
	test -e $@ && touch $@
