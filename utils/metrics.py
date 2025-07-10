import numpy as np
import itertools
import tomotopy as tp
def calculate_perplexity(model, corpus):
    topic_dist, ll = model.infer(corpus, together=True)
    word_dist = np.array([len(doc.words) for doc in topic_dist])
    perplexity_score = np.exp2(-np.sum(ll)/np.sum(word_dist))
    return perplexity_score

def calculate_coherence(model, topic_reference_corpus, preset):
    topics = itertools.chain(*((w for w, _ in model.get_topic_words(k, top_n=10)) for k in range(model.k)))
    coh = tp.coherence.Coherence(corpus = topic_reference_corpus, coherence=preset, targets = topics)
    coherence = []
    for k in range(model.k):
        coherence.append(coh.get_score(words = (w for w, _ in model.get_topic_words(k, top_n=10))))
    average_coherence_score = sum(coherence) / len(coherence)
    return average_coherence_score