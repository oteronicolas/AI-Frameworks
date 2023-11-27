from methods import create_df
from multiprocessing import freeze_support

def main():
    freeze_support()  # need this line for Windows multiprocessing
    dico = {'df': create_df('./extract_data/MLP-20M'),
            'n_reco': 5,
            'index': 16}
    print(dico['df'].head())
    print(dico)

if __name__ == '__main__':
    main()
