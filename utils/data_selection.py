import time
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing as mp
from utils.preprocesslib import Preprocess

warnings.filterwarnings('ignore')

    
class Hyperexponential:
    def __init__(self, data: np.ndarray | list, n: int) -> None:
        """_summary_

        Args:
            data (np.ndarray | list): The data points that needs to be fitted to hyperexponential
            n (int): The number of mixtures in hyperexponential distribution
            distribution_type (str): The distribution type of hyperexponential (possible arg: 'discrete' or 'continuous')
        """
        self.data = data
        self.n = n
    
    @staticmethod
    def pdf(data: np.ndarray | list, weights: np.ndarray | list, rates: np.ndarray | list):
        return weights[:, None] * (np.exp(rates)[:,None] - 1) * np.exp(np.dot(-rates[:, None], data[None, :]))
        
    @staticmethod
    def cdf(data: np.ndarray | list, weights: np.ndarray | list, rates: np.ndarray | list):
        return 1 - weights[:,None] * np.exp(np.dot(-rates[:, None], data[None, :]))

    def _initialize_parameters(self):
        """ Initialize the weights uniformly and rates based on quantiles. """
        return np.full(self.n, 1/self.n), 1 / np.quantile(self.data, np.linspace(1/self.n, 1, self.n))

    def _e_step(self, weights: np.ndarray | list, rates: np.ndarray | list):
        """ Expectation step: calculate responsibilities (posterior probabilities). """
        weighted_densities = self.pdf(self.data, weights, rates)
        return weighted_densities / weighted_densities.sum(axis=0)

    def _m_step(self, responsibilities: np.ndarray | list):
        """ Maximization step: update parameters based on responsibilities. """
        return responsibilities.mean(axis=1), np.sum(responsibilities, axis=1) / np.sum(responsibilities * self.data, axis=1)

    def EM_fit(self, max_iter: int=5000, tol: float=1e-320):
        weights, rates = self._initialize_parameters()
        for _ in range(max_iter):
            old_rates = rates.copy()
            
            # E-step
            responsibilities = self._e_step(weights, rates)
            
            # M-step
            weights, rates = self._m_step(responsibilities)
            
            # Check convergence
            if np.all(np.abs(rates - old_rates) < tol):
                break
        
        return weights, rates

class DocumentSelection:
    def __init__(self, doc_list):
        preprocess = Preprocess()
        self.k = None
        self.doc_list = doc_list
        self.word_frequency_vectors = preprocess.word_frequency_builder(doc_list)
        self.selected_doc_hyper = None
        self.hyper_std = []


    def well_form_topic_hyper(self, freq, k):
        hyperexponential = Hyperexponential(freq, k)
        pi, lamda = hyperexponential.EM_fit()
        empirical = np.unique(freq, return_counts=True)[1]
        empirical = empirical / empirical.sum()
        
        predicted = hyperexponential.pdf(np.unique(freq), pi, lamda).sum(axis=0)
        predicted = predicted / predicted.sum()
        
        result = (((predicted - empirical) ** 2).sum() / len(freq)) ** 0.5
        
        return result
    
    def _fitting_hyper(self, frequency):
        if len(frequency) == 0:
            res = -1.
        else:
            res = self.well_form_topic_hyper(frequency, self.k)
            
        return res


    def document_selector_hyper(self, k: int=3):
        self.k = k
        start = time.time()
        with Pool(mp.cpu_count()) as p:
            results = p.map(self._fitting_hyper, self.word_frequency_vectors)
        
        for result in results:
            self.hyper_std.append(result)
        end = time.time()
        print('took:', end - start)
        
    
    def select_hyper_document(self, threshold: float, save: bool = False):
        selected_index = (np.array(self.hyper_std) <= threshold) & (np.array(self.hyper_std) >= 0.)
        
        choosen_docs_hyper = list()
        for i, boolean in enumerate(selected_index):
            if boolean:
                choosen_docs_hyper.append(self.doc_list[i])
            
        self.selected_doc_hyper = choosen_docs_hyper
        
        if save:
            self._save_docs('_hyper', self.selected_doc_hyper)

    

    def _save_docs(self, filename, suffix, choosen_documents):
        dataframe = pd.DataFrame(choosen_documents, columns = ['texts'])
        dataframe.to_csv(filename + suffix + '.csv', index = False)
