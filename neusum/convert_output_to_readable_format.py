if __name__ == '__main__':
    import os
    dir = '../model_output'
    f_nyt = '0.456,0.253,0.383-nytTrue5.0-1True5-17218-cp_0.55'
    f_cnn = '0.327,0.122,0.290-cnnTrue1.0-1True3-1093-cp_0.5'
    f_dm = '0.427,0.192,0.388-dmTrue10.0-1True3-10397-cp_0.7'
    import pickle
    with open(f_nyt, 'rb') as fd:
        lines = fd.read().splitlines()