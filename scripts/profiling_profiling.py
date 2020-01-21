# lib imports

import csv
import nltk
import itertools
import datetime
# spacy import and text tokenization function
import spacy
nlp = spacy.load('en')


def sentence_tokenize(text):
    doc = nlp(text)
    return [sent.string.strip() for sent in doc.sents]


# NLTK text tokenization and calculation of computation time:&lt;/pre&gt;
def run_nltk():
    print("Reading CSV file...")
    sample_str = []
    # record time stamp before 
    # tstart = datetime.now()
    with open('../data/reddit-comments-2015-08.csv', 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # read is an iterator which contains lines which can be itered through an iterator.
        next(reader)
        # iterate through all the lines and tokenize the text. 
        for row in reader:
            sample_str = row[0]
            tokzd = nltk.sent_tokenize(sample_str.lower()) # text tokenization through NLTK's function
            sentences = itertools.chain(tokzd) 
    # record time stamp afterwards
    # tend = datetime.now()
# print time took to tokenize the text
# print ( tend - tstart )
# Output : 0:00:01.961066
# Spacy text tokenization and calculation of computation time: 
# record time stamp before 
# tstart = datetime.now()


def run_spacy(): 
    with open('reddit-comments-2015-08.csv', 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # read is an iterator which contains lines which can be itered through an iterator.
        next(reader)   
    # iterate through all the lines and tokenize the text.
        for row in reader:
            sample_str = row[0]
            tokzd = sentence_tokenize(sample_str.lower())
            sentences = itertools.chain(tokzd)
    # record time stamp afterwards
    # tend = datetime.now()
    # print time took to tokenize the text
    # print ( tend - tstart )
    # Output : 0:02:58.788951


if __name__ == '__main__':
    run_spacy()
