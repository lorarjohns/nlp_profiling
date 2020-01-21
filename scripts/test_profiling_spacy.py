from profiling_spacy import nlp, clean_text, nltk_clean_nlp, get_sample_docs, list_pipe_docs, pipe_docs, get_piped_docs
# import numpy
import pytest
import spacy
examples = ['Hello, my name is Inigo Montoya', 'You killed my father.', 'Prepare to die']

def test_nospacy_clean():
    doc_example = '''Let\s try some NER: Barack Obama, Germany, $5 million, ten o\'clock.
                 the parts of speech get lemmatized differently: it\'s fun making things, 
                 the making of the film, making fun of you'''
    ans = clean_text(doc_example, spacy=False)
    assert ans == "let try ner barack obama germany 5 million ten clock part speech get lemmatized differently fun make thing make film make fun"


def test_spacy_clean():
    doc_example = "Let\'s try some NER: Barack Obama, Germany. it\'s fun making things, the making of the film, making fun of you"
    ans = clean_text(nlp(doc_example), spacy=True)
    assert ans.text == "let try ner Barack Obama Germany fun make thing making film make fun"


def test_clean_text_type():
    doc_example = "You keep using that word. I do not think it means what you think it means."
    with pytest.raises(AttributeError):
        clean_text(doc_example, spacy=True)


def test_return_spacy():
    doc_example = examples
    ans = list_pipe_docs(doc_example)
    assert type(ans[0]) == spacy.tokens.doc.Doc


def test_return_nltk():
    doc_example = examples
    ans = nltk_clean_nlp(doc_example)
    assert type(ans[0]) == str


def test_spacy_list():
    doc_example = examples
    ans = list_pipe_docs(doc_example)
    assert ans[0].__class__ == spacy.tokens.doc.Doc


def test_spacy_generator():
    pass


def test_spacy_pipe():
    pass


def test_nltk():
    pass
