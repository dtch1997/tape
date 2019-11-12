# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:33:40 2019

@author: Daniel
"""




"""
The APIs I need are as follows: 
"""

from typing import List

class DataGenerator(object):
    
    def __init__(self,
                 datafiles: List[str],
                 config = {} # Put in whatever other arguments you need, e.g. how to assign train/test
                 ):
        """
        Split the 5 labelled datasets (deterministically) into 4 training + 1 test. 
        Within each training dataset, randomly do an 80/20 train/val split. 
        """
        raise Exception("Not implemented yet")

    def sample_batch(batch_type, # meta_train, meta_val, meta_test
                     batch_size_u,
                     batch_size_l,
                     sample_config={} # Put in whatever other arguments you need, e.g. which dataset to select from
                    ):

        """
        meta_train: 
            (Deterministically) pick 1 out of 4 training datasets
            sample (batch_size_u) unlabelled sequences from the unlabelled set. 
            sample (batch_size_l) labelled sequences from the training dataset
            Bonus points if your implementation guarantees seeing all training data eventually. 
            
        meta_val:
            (Deterministically) pick 1 out of 4 training datasets 
            sample (batch_size_l) labelled sequences from the validation dataset
            Bonus points if your implementation guarantees seeing all validation data eventually
            
        meta_test:
            sample (batch_size_l) labelled sequences from the labelled test dataset
            Bonus points if your implementation guarantees seeing all test data eventually 
        """
        raise Exception("Not implemented yet")



"""
The code below is a suggestion of how you could decompose your code.
Feel free to modify it if you think it'll be faster / easier / better. 
"""

from pathlib import Path

def get_filepath(filename):
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    return filepath

class Dataset(object):
    """
    Responsible for loading data from one dataset.
    """
    
    def __init__(self, 
                 datafile: str,
                 is_training: bool):
        self._datapath = get_filepath(datafile)
        self._is_training = is_training
    
    def sample_batch():
        pass
    

    
        
    
    
    
    