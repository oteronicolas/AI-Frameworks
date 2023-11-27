import gradio as gr
from PIL import Image
import requests
import io

def similar_poster(image):
    # Conversion de l'image en données binaires
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    api_url = "http://df63-104-196-8-77.ngrok.io/upload"

    # Utilisez le nom de fichier "image" pour envoyer l'image à l'API Flask
    response = requests.post(api_url, data=img_binary.getvalue())
    print(response)


    if response.status_code == 200:
        similar_image_paths = response.json()["similar_poster"]
        similar_images = [Image.open(path) for path in similar_image_paths]
        all_images =  similar_images

        return all_images[0], all_images[1], all_images[2], all_images[3],all_images[4]


if __name__ == '__main__':
    gr.Interface(
        fn=similar_poster,
        inputs=gr.Image(type="pil", label="Choose an image file"),
        outputs=["image","image","image"],
        live=True,
        description="Select an image file and view it along with 5 similar images.",
    ).launch(debug=True, share=True)