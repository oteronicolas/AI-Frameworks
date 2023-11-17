import zipfile
import os

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



zip_file_path = [r'C:\Users\alexa\Desktop\projet aif\movies_metadata.csv.zip',
                 r'C:\Users\alexa\Desktop\projet aif\ratings.csv.zip']
extraction_directory = './extract_data/'

unzip_file(r'C:\Users\alexa\Desktop\projet aif\movielens-20m-posters-for-machine-learning.zip', extraction_directory)