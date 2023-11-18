import zipfile
import os

path="./"
def unzip_file(zip_file, extract_dir):

    if not os.path.exists(zip_file):
        print(f"The file '{zip_file}' does not exist.")
        return

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Successfully extracted '{zip_file}' to '{extract_dir}'.")
    except zipfile.BadZipFile as e:
        print(f"Error: Bad ZIP file - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

extraction_directory = './extract_data/'

unzip_file('./movielens-20m-posters-for-machine-learning.zip', extraction_directory)