import gradio as gr
from PIL import Image
import gradio as gr
import gradio as gr
from PIL import Image
import requests
import io
import json
import pandas
import os

def traiter_texte(method,texte):
    # Fonction pour traiter le texte
    # Remplacez cela par votre code de traitement de texte
    return f"Texte traité: {texte}"

"""
def similar_poster(methode,texte):
    #df = create_df('./extracted_data//movielens-20m-posters-for-machine-learning/MLP-20M')
    df = pandas.read_pickle("df.pkl")
    #index_movie = 15
    image_features = extract_features_from_image(methode,texte)
    #dico = {'nb_reco': 5,
    #        'vec': df['features'].iloc[index_movie].tolist()}
    dico = {'nb_reco': 5, 'vec': image_features.tolist()}
    jsonData = json.dumps(dico)
    headers = {'Content-Type': 'application/json'}

    response = requests.post("http://127.0.0.1:5000/predict", data=jsonData, headers=headers)
    pred = df.iloc[response.json()["prediction"]]
    paths=pred.path
    paths = [path.replace("\\", "/")for path in paths]
    print(paths)
    if response.status_code == 200:
        similar_images = [Image.open(path) for path in paths]
        all_images =  similar_images

        return all_images[0], all_images[1], all_images[2], all_images[3],all_images[4]

"""


global_interface = gr.Interface(
    fn=traiter_texte,
    inputs=[gr.Radio(["Bag_of_words", "Word2Vec","BERT"], label="Choisissez le type d'entrée"),gr.Textbox()],
    outputs="text"
)

global_interface.launch(debug=True, share=True)
