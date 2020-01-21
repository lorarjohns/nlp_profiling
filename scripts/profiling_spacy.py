# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#from IPython import get_ipython

# %% [markdown]
# # Profiling NLP: optimizing code with spaCy
# 
# ## Objective
# 
# To compare two Python natural-language processing libraries with respect to their speed and efficiency
# 
# ## Architecture
# 
# NLTK is very popular but not optimized for speed. SpaCy is built with Cython, which gives it an enormous speed advantage, but many people misuse it in a way that slows it down to Python speeds. We'll examine ways to achieve common NLP tasks while avoiding time overhead.
# 
# ## Tasks
# 
# The processing pipeline(s) include:
# 
#     NLTK version:
#       - tokenize: split texts into individual tokens
#       - lowercase: normalize the vocabulary by case
#       - stopword removal: remove tokens if they appear in a specified list
#       - tag: tag part of speech (for lemmatization)
#       - lemmatize: normalize the vocabulary to the base form of each token
#       - join (optional): return the list of tokens in one joined string
#       
# The standard spaCy pipeline includes all of this, plus dependency parsing and NER:
#       
#     spaCy version:
#       - tokenize
#       - tag: tag part of speech
#       - parse: perform dependency parsing
#       - named entity recognition: extract named entities according to statistical model
#       - lemmatize
#       - join (optional)
# 

# %%
import pandas as pd
# import numpy as np
import spacy
# import nltk
# import gensim
# from itertools import dropwhile
import requests
import zipfile
import io
import logging
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
# from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from collections import defaultdict


# %%
# get_ipython().run_line_magic('load_ext', 'line_profiler')
logger = logging.getLogger(__name__)


# %%
stop_words = stopwords.words('english')



# %%
def clean_text(doc, spacy=True, printed=False, list_tokens=False):
    '''
    define a simple preprocessing pipeline for general NLP tasks.
    non-spaCy version:
      - tokenize: split texts into individual tokens
      - lowercase: normalize the vocabulary by case
      - stopword removal: remove tokens if they appear in a specified list
      - tag: tag part of speech (for lemmatization)
      - lemmatize: normalize the vocabulary to the base form of each token
      - join (optional): return the list of tokens in one joined string
      
      the spaCy pipeline includes all of this, plus dependency parsing and NER:
      
    spaCy version: (we will disable some of these pipes in testing.)
      - tokenize
      - lemmatize
      - stopword removal
      - tag: tag part of speech
      - parse: perform dependency parsing
      - named entity recognition: extract named entities according to statistical model
      - join (optional)
    '''  

    if spacy:

        try:
            doc = [token.lemma_ for token in doc if not token.is_stop and not token.pos_ in ['PRON', 'PUNCT']]
            #dropwhile(lambda x: not (x.is_stop and x.pos_ in ['PRON', 'PUNCT']), tokens)
            #if not token.is_stop and not token.pos_ in set(['PRON', 'PUNCT'])
            if not list_tokens:
                return nlp.make_doc(' '.join([token for token in doc]))
            else:
                return doc
        except AttributeError as ae:
            print(f'''ERROR! if parameter spacy == True, corpus input must be of type spacy.tokens.doc.Doc, not {doc.__class__}!\ne.g.: clean_text(nlp("You keep using that word. I do not think it means what you think it means.")''')
            logger.error(ae)
            raise
    else:
        '''
        spaCy's default pipeline includes tokenizer + lemmatizer + POS-tagging
        we've added stopword removal to both processes
        '''
        def pos_tag_nltk(token, printed=False):
            tag_map = defaultdict(lambda : wn.NOUN)
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV
        
            nonlocal lemmatizer
        
            token, tag = zip(*pos_tag([token]))
            lemma = lemmatizer.lemmatize(token[0], tag_map[tag[0][0]])
            if printed:
                print(token[0], "=>", lemma)
            return lemma

        new_doc = []
        tokenizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()
        doc = tokenizer.tokenize(doc)
        for token in doc:
            if token.lower() not in stop_words:
                new_doc.append(pos_tag_nltk(token.lower(), printed))
        if not list_tokens:
            return ' '.join([token for token in new_doc])
        else:
            return new_doc


# %%
doc_example = '''Let\'s try some NER: Barack Obama, Germany, $5 million, ten o\'clock.
the parts of speech get lemmatized differently: it's fun making things, the making of the film, making fun of you'''

alt_example = '''This is a Doc. I have made one, and it\'s clean now! Pretty cool, huh? It\'s fun making docs in spaCy. 
               Let\'s try for some NER: Barack Obama, Germany, $5 million, Quakers, ten o\'clock.
               Did you notice that made and make are stop words in spaCy but making isn't?
               the parts of speech get lemmatized differently: it's fun making things, the making of the film, making fun of you
               WordNet treats lemmas differently: repeat repeating repeats repeated repetition repetitive
               in spaCy, clocks are different from o\'clock '''

'''
Note the differences in spaCy's stopwords list
and lemmatization vs. NLTK's. Also note that
spaCy won't eliminate newline characters for you.
'''

nlp = spacy.load('en_core_web_lg')
nltk_ex = clean_text(doc_example, spacy=False)
spacy_ex = clean_text(nlp(doc_example))

print('Example of stopword removal and lemmatization')
print('')
print(f'Unprocessed: {doc_example}')
print(f'NLTK: {nltk_ex}')
print('')
print(f'spaCy: {spacy_ex}')


# %%
'''
However, spaCy's conservative lemmatization 
lets it keep the entities that it recognizes together,
unlike NLTK, which does not keep them in one unit.

spaCy also takes part of speech into account when lemmatizing.
if syntactic nuance is important to your use case, this
can be a valuable disambiguating tool.

NLTK: ten, clock
spaCy: ten o'clock

NLTK:  making of + N (nominalized V) -> make
       making + N (V, progressive aspect) -> make
spaCy: making of + n -> making (retains form of nominalization)
       making + N -> make 

NB: it's sensitive enough to know the difference between 
"making of + N" and "making N of"

'''
spacy_doc = nlp(doc_example)
spacy_doc.ents

print('Example of polysemy with spaCy:')
print(clean_text(nlp(u'The making of the film')))
print(clean_text(nlp(u'making a banana cream pie')))
print(clean_text(nlp(u'making fun of your dumb photos')))
print(clean_text(nlp(u'making light of the situation')))


# %%
'''
note that the order in which pipeline components are used counts.
this ordering recognizes two kinds of Barack Obama,
because he has been lowercased.
'''

nlp = spacy.load('en_core_web_lg')
print(f'Default spaCy pipeline: {nlp.pipeline}')

docs = ['doc one', 'barack obama', 'fifty billion', 'Barack Obama']


def nltk_clean_nlp(docs, spacy=False):
    new_docs = []
    for doc in docs:
        clean_text(doc, spacy=False)
        new_docs.append(doc)
        if spacy:
            doc = nlp(doc)
            ents = doc.ents
            print(doc, ents)
    return new_docs


print('Example of lowercase vs. uppercase entities')
print(nltk_clean_nlp(docs, spacy=True))


# %%
# %%

def get_sample_docs(url='http://archives.textfiles.com/media.zip'): # http://archives.textfiles.com/anarchy.zip
    '''Get some text files for testing.'''
    print('Fetching sample documents to test on.')
    print('...')
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    prof_docs = []
    for file in z.namelist():
        tmp = z.extract(file)
        try:
            with open(file, 'r') as f:
                prof_docs.append(f.read())
        except Exception as e:
            print(f'could not read {file}: {e}')
    return prof_docs


# %%
def fetch():
    '''Download the documents for testing'''
    prof_docs = get_sample_docs()
    print(f'Number of test docs in corpus: {len(prof_docs)}')

# %% [markdown]
# # Profiling

# ## Round 1: NLTK

# ### Total time: 1897.58 s (that's 31.6 minutes)
# ### 60.4%: clean_text
# ### 39.1%: nlp(doc)

# %%
# get_ipython().run_line_magic('lprun', '-f nltk_clean_nlp -f clean_text -T nltk_pipe.txt nltk_clean_nlp(prof_docs)  # -T nltk_pipe.txt')

# %%


nlp = spacy.load('en_core_web_lg')
print(f'Current spaCy pipeline: {nlp.pipeline}')


def list_pipe_docs(docs):
    '''Disable ner, parsing, and test the python-list conversion of spaCy\'s generator.'''
    # assert len(nlp.pipeline) == 3, 'Did you forget to pass the plain nlp object in?'

    all_docs = []  
    for doc in list(nlp.pipe(docs, disable=['ner', 'parser'])):
        new_doc = clean_text(doc)
        all_docs.append(new_doc)

    return all_docs

# %%
#get_ipython().run_line_magic('lprun', '-f list_pipe_docs -f clean_text -T spacy_list.txt list_pipe_docs(prof_docs)')

# %% [markdown]
# # Round 3: spaCy, using (some) Cython optimization
# 
# ### Total time:

# %%


nlp_pipeline = spacy.load('en_core_web_lg')
nlp_pipeline.add_pipe(clean_text, 'clean_text')
print(f'Augmented spaCy pipeline: {nlp_pipeline.pipeline}')


def pipe_docs(docs):
    for doc in nlp_pipeline.pipe(docs):
        yield doc


def get_piped_docs(docs):
    my_docs = []
    for doc in pipe_docs(docs):
        my_docs.append(doc)
    return my_docs


# %%

#get_ipython().run_line_magic('lprun', '-f clean_text pipe_docs(docs)')

'''
clean_text is much faster in the pipe_docs function
the slowest part is the string join to return a spacy doc at the end
'''

# %%

#get_ipython().run_line_magic('lprun', '-f pipe_docs -f clean_text pipe_docs(prof_docs)')

# %% [markdown]
# # The Winner: spaCy pipeline
# 
# * spaCy with Cython: 11 minutes
# * spaCy with Python lists: 14.5 minutes
# * NLTK: 31.6 minutes
# 
# The slowest parts of the pipeline are still the Python parts: the list comprehension (23.6% of total time, 2821.8 ms) and the string join with list comprehension (76.3% of total time, 9118.1 ms). (This could be further optimized, too.)
# 
# 
# ### Takeaways
# 
# * NLTK is a significantly slower way to preprocess text
# * Using spaCy's nlp in a for-loop is still better than NLTK, but not as good as nlp.pipe()
# * However, converting the nlp.pipe object to a Python list negates several minutes of time saved due to Cython efficiencies
if __name__ == "__main__":
    import datetime
    reddit = pd.read_csv('reddit-comments-2015-08.csv')
    blog_docs = reddit.body.tolist()
    # start a timer
    start = datetime.now()

    # TEST NLTK
    nltk_test = nltk_clean_nlp(blog_docs)
    nltk_end = datetime.now()-start
    print(f'NLTK done: {nltk_end}')

    # TEST SPACY (LIST)
    spacy_list_test = list_pipe_docs(blog_docs)
    sl_end = datetime.now()-start
    print(f'spaCy with list done: {sl_end}')

    # TEST SPACY (GENERATOR)
    spacy_pipe_test = get_piped_docs(blog_docs)
    sg_end = datetime.now()-start
    print(f'spaCy with generator done: {sg_end}') 