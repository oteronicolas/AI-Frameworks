import gradio as gr
from PIL import Image
from methods import create_df
from methods import extract_features_from_image
import requests
import io
import json
import pandas

def similar_poster(image):
    #df = create_df('./extracted_data//movielens-20m-posters-for-machine-learning/MLP-20M')
    df = pandas.read_pickle("df.pkl")
    #index_movie = 15
    image_features = extract_features_from_image(image)
    #dico = {'nb_reco': 5,
    #        'vec': df['features'].iloc[index_movie].tolist()}
    dico = {'nb_reco': 5, 'vec': image_features.tolist()}
    jsonData = json.dumps(dico)
    headers = {'Content-Type': 'application/json'}

    response = requests.post("http://127.0.0.1:5000/predict", data=jsonData, headers=headers)
    pred = df.iloc[response.json()["prediction"]]
    paths=pred.path
    print(paths)
    if response.status_code == 200:
        similar_images = [Image.open(path) for path in paths]
        all_images =  similar_images

        return all_images[0], all_images[1], all_images[2], all_images[3],all_images[4]


if __name__ == '__main__':
    gr.Interface(
        fn=similar_poster,
        inputs=gr.Image(type="pil", label="Choose an image file"),
        outputs=["image","image","image","image","image"],
        live=True,
        description="Select an image file and view it along with 5 similar images.",
    ).launch(debug=True, share=True)