# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:08:18 2019

@author: Daniel
"""

from tensorflow.keras import Model
import HypersphericalLoss as HL

class HypersphericalModel(Model):
    
    def __init__(self, 
                 n_symbols: int,
                 encoder,
                 loss
                 ):
        """
        Args:
            prototypes: [n_prototypes, embed_dim]
            encoder: Encodes a sequence of tokens into a vector of shape [embed_dim]
        """
        super().__init__()
        self.enc = encoder
        
    def call(self, xu, xl, pl, loss):
        """
        Semi-supervised loss function
        
        Args:
            xu: [U, ?] unlabelled sequences
            xl: [L, ?] labelled sequences
            yl: [L, prototype_dim] prototypes
        """
        zu = self.enc.encode(xu) # (U, embed_dim)
        # Generate prototypes
        zl = self.enc.encode(xl) # (L, embed_dim)
        pu = HL.get_fake_prototypes(preds_l = zl, prototypes_l = pl, preds_u = zu)
        return loss(zu, pu) + loss(zl, pl)
        
        
        
        
        