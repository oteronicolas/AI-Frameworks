import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from tqdm.notebook import tqdm


#def get_index(path,df):
#   return index

def create_df():
    class ImageAndPathsDataset(datasets.ImageFolder):
        def __getitem__(self, index):
            img, _ = super(ImageAndPathsDataset, self).__getitem__(index)
            path = self.imgs[index][0]
            return img, path
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    dataset = ImageAndPathsDataset('./extracted_data/movielens-20m-posters-for-machine-learning/MLP-20M', transform)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)
    x, paths = next(iter(dataloader))
    mobilenet = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
    model = mobilenet
    model.classifier = nn.Flatten()
    features_list = []
    paths_list = []

    for x, paths in tqdm(dataloader):
        with torch.no_grad():
        embeddings = model(x)
        features_list.extend(embeddings.numpy())
        paths_list.extend(paths)

        df = pd.DataFrame({
            'features': features_list,
            'path': paths_list
            })
    return df

def recommandations(idx,n_reco):
    df= pd.read_pickle('embeddings_dataframe.pkl')
    features = np.vstack(df['features'])
    cosine_sim = cosine_distances(features)
    img=df['path'][idx]
    recos = cosine_sim[idx].argsort()[1:n_reco]
    recos_path=df['path'][recos]
    return recos_path
