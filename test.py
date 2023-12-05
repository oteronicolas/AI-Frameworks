from methods import create_df
from multiprocessing import freeze_support
from torchvision import datasets


class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _ = super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path
def main():
    import torch
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

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
    dataset = ImageAndPathsDataset('./extracted_data//movielens-20m-posters-for-machine-learning/MLP-20M', transform)

    dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)
    import torchvision.models as models
    mobilenet = models.mobilenet_v3_small(weights='DEFAULT')
    import torch.nn as nn
    mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights)
    model = mobilenet
    model.classifier = nn.Flatten()
    import pandas as pd
    from tqdm.notebook import tqdm

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
    import os
    current_dir = os.getcwd()
    pickle_file_path = 'df.pkl'
    df.to_pickle(f'{current_dir}+{pickle_file_path}')

if __name__ == '__main__':
    main()
