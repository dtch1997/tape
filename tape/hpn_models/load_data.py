# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:33:40 2019

@author: Daniel
"""




"""
The APIs I need are as follows: 
"""

from typing import List, Dict
import os
import re


import numpy as np
import tensorflow as tf

from tape.tasks.BeplerLanguageModelingTask import BeplerLanguageModelingTask
from tape.tasks.BeplerContactMapTask import BeplerContactMapTask
from tape.tasks.ContactMapTask import ContactMapTask
from tape.tasks.StabilityTask import StabilityTask
from tape.tasks.FluorescenceTask import FluorescenceTask
from tape.tasks.LanguageModelingTask import LanguageModelingTask
from tape.tasks.MaskedLanguageModelingTask import MaskedLanguageModelingTask, mask_params
from tape.tasks.NetsurfTask import NetsurfTask
from tape.tasks.BeplerPairedScopeTask import BeplerPairedScopeTask
from tape.tasks.RemoteHomologyTask import RemoteHomologyTask
from tape.tasks.SecondaryStructureTask import SecondaryStructureTask, secondary_structure_params
from tape.tasks.UnidirectionalLanguageModelingTask import UnidirectionalLanguageModelingTask
class DataGenerator(object):
    
    def __init__(self,
                 batch_size_u,
                 batch_size_l,
                 tasks=[(StabilityTask, 'stability', True, False),
                        (FluorescenceTask, 'fluorescence', True, False),
                        (SecondaryStructureTask, 'secondary_structure', False, True),
                        (RemoteHomologyTask, 'remote_homology', False, False),
                        (LanguageModelingTask, 'pfam', False, False),
                        (ContactMapTask, 'proteinnet', False, False)] # (task, id) tuple list
                 ):
        """
        Split the 5 labelled datasets (deterministically) into 4 training + 1 test. 
        Within each training dataset, randomly do an 80/20 train/val split. 
        """
        self.tasks = tasks
        self.batch_size_u = batch_size_u
        self.batch_size_l = batch_size_l
        self.meta_train = []
        self.meta_val = []
        self.meta_test = []

        for t, data_folder in tasks:
            t = t()
            train_files = t.get_train_files('data/')
            valid_files = t.get_valid_files('data/')

            train_data = (tf.data.TFRecordDataset(train_files)
                            .map(t._deserialization_func, num_parallel_calls=64)
                            .shuffle(100, reshuffle_each_iteration=True)
                            .batch(self.batch_size_l))
            valid_data = (tf.data.TFRecordDataset(valid_files)
                            .map(t._deserialization_func, num_parallel_calls=64)
                            .shuffle(100, reshuffle_each_iteration=True)
                            .batch(self.batch_size_l))
            self.meta_train.append(train_data)
            self.meta_val.append(valid_data)

        # Last Two are held out task and Unlabelled task

        self.meta_test.append( (self.meta_train.pop(), self.meta_val.pop()) )

        # Reset batch sizes for the unlabelled data
        self.ul_data_train = self.meta_train.pop().flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)).batch(self.batch_size_u)
        self.ul_data_val = self.meta_val.pop().flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)).batch(self.batch_size_u)


        '''
        # Create Iterators for the Datasets
        self.meta_train_iters = [iter(d) for d in self.meta_train]
        self.meta_val_iters = [iter(d) for d in self.meta_val]
        self.meta_test_iters = [iter(d) for d in self.meta_test]

        self.ul_train_iter = iter(self.ul_data_train)
        self.ul_val_iter = iter(self.ul_data_val)
        '''


    def sample_batch(self,
                     batch_type, # meta_train, meta_val, meta_test
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
        
        Return: (task_id, is_regression, labelled_sample, unlabelled_sample)
            task_id: A unique integer / string corresponding to the task. Will be used as a dictionary key. 
            is_regression: True if it's a regression task, False otherwise. 
            is_per_amino_acid: True if there's one label for each amino acid, False otherwise
            labelled_sample: Labelled dataset. 
            unlabelled_sample: None if not relevant. 
        """

        ul_sample = None
        dataset = None
        selector = int(np.random.randint(0,4))
        task_id = None
        if batch_type == 'meta_train':
            dataset = self.meta_train[selector]
            ul_sample = next(iter(self.ul_data_train))
            task_id, is_regression, is_per_aa = self.tasks[selector][1:]

        elif batch_type == 'meta_val':
            dataset = self.meta_train[selector]
            task_id, is_regression, is_per_aa = self.tasks[selector][1:]
        # must be meta_test !!!!
        elif batch_type == 'meta_test':
            dataset = self.meta_test[0][0]
            task_id, is_regression, is_per_aa = self.tasks[-1][1:]
        else:
            raise NotImplementedError('No other case for batch_type')

        sample = next(iter(dataset))
        
        

        return task_id, is_regression, is_per_aa, sample, ul_sample
