import sys
import os
sys.path.insert(0,os.path.dirname(__file__))
import pickle as pi
import pandas as pd
import random
import infrastructure.generative_models.GAN.gan_lstm_refactoring.scripts.model as KOSTIL_FOR_PICKL
sys.modules['scripts.model'] = KOSTIL_FOR_PICKL
sys.modules['scripts.utils'] = KOSTIL_FOR_PICKL
sys.modules['scripts.layers'] = KOSTIL_FOR_PICKL
sys.modules['scripts.tokenizer'] = KOSTIL_FOR_PICKL


def generate(n):
    gan_mol = pi.load(open('infrastructure/generative_models/GAN/gan_lstm_refactoring/weights/v4_gan_mol_124_0.0003_8k.pkl', 'rb'))
    
    # generate smiles molecules
    smiles_list = gan_mol.generate_n(n)
    return smiles_list
if __name__=='__main__':
    print(generate(4))
    