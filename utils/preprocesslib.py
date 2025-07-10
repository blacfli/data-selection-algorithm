# Importing libraries
import re
import sys
import spacy
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from tomotopy.utils import Corpus





class BasePreprocess:
    def __init__(self) -> None:
            # NLTK Stop words
        pass
    
    def _init_preprocess(self, data):
        documents = data.dropna(how='all')
        # Remove punctuation
        documents['paper_text_processed'] = documents['texts'].map(lambda x: re.sub('[,\.!?]', ' ', x))

        # Convert the titles to lowercase
        documents['paper_text_processed'] = documents['paper_text_processed'].map(lambda x: x.lower())
        
        documents = documents.paper_text_processed.values.tolist()

        return documents
    
    
    def _sent_to_words(self, sentences):
        for sentence in sentences:
            yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def _remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words] for doc in texts]

    def _make_bigrams(self, data_words, texts):
        bigram = Phrases(data_words, min_count=5, threshold=100)
        bigram_mod = Phraser(bigram)
        return [bigram_mod[doc] for doc in texts]
    
    def word_frequency_builder(self, doc):
        data_words_nostops = self._remove_stopwords(doc)
        dictionary = Dictionary()
        BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in data_words_nostops]

        word_frequencies = []
        for corp in BoW_corpus:
            word_freq = np.sort(np.array([count for _, count in corp]))[::-1]
            word_frequencies.append(word_freq)
        
        return word_frequencies

    def _lemmatization(self, texts, lemma_model, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        nlp = spacy.load(lemma_model, disable=['parser', 'ner'])
        texts_out = []
        for sent in tqdm(texts, file = sys.stdout, ascii = ' >='):
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out


    def _document_preprocessing(self, papers, lemma_models):
        lemma_model_tag = 'en_core_web_trf'
        if isinstance(papers, pd.DataFrame):
            data = self._init_preprocess(papers)
        else:
            data = papers
        
        data_words = list(self._sent_to_words(data))

        # Remove Stop Words
        print('=== Removing Stop Word ===')
        data_words_nostops = self._remove_stopwords(data_words)

        # Form Bigrams
        print('=== Making Bigrams ===')
        data_words_bigrams = self._make_bigrams(data_words, data_words_nostops)

        # Do lemmatization keeping only noun, adj, vb, adv
        print('=== Lemmatizing Word ===')
        if lemma_models == "efficient":
            lemma_model_tag = 'en_core_web_sm'
        
        data_lemmatized = self._lemmatization(data_words_bigrams, lemma_model_tag, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        return data_lemmatized

class Preprocess(BasePreprocess):
    def __init__(self):
        spacy.require_gpu()
        extended_stop_word_list = ['from', 'subject', 're', 'edu', 'use', 'say', 'says','said', 'smith', 'temporao', 'arsenio',
                            'arthur', 'bancorp', 'birr', 'daniel', 'joseph', 'lawrence', 'miller', 'merril', 'lynch', 'pierce',
                            'fenner', 'william', 'wilson', 'iv', 'inc', 'tex', 'woodco', 'ct', 'dougla', 'elgin', 'howard',
                            'larsen', 'kenneth', 'ryan', 'forman', 'brent', 'casey', 'edmund', 'hess', 'john', 'muskie', 'reagan',
                            'schneider', 'scowcroft', 'shultz', 'stephen', 'weinberger', 'reuter', 'co', 'also']
        
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(extended_stop_word_list)
    
    def _make_training_corpus(self, _ori_doc, lemma_model):
        _lemmas = [ele for ele in self._document_preprocessing(_ori_doc, lemma_model) if ele != []]

        _corpus = Corpus()

        for lemma in _lemmas:
            _corpus.add_doc(lemma)

        del _lemmas

        return _corpus

    def make_training_corpora(self, _ori_doc, _selected_doc, lemma_model):
        reference_corpus = self._make_training_corpus(_ori_doc, lemma_model)
        model_selected_corpus = self._make_training_corpus(_selected_doc, lemma_model)

        return (reference_corpus, model_selected_corpus)

