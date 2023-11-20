from methods import create_df
from multiprocessing import freeze_support

def main():
    freeze_support()  # need this line for Windows multiprocessing
    dico = {'df': create_df('./extracted_data/movielens-20m-posters-for-machine-learning/MLP-20M'),
            'n_reco': 5,
            'index': 16}
    print(dico['df'].head())
    print(dico)

if __name__ == '__main__':
    main()
