# SusGen Project for A* and NUS
# 2024-03-15: Xuan W.

import os
import csv
import torch
import numpy as np
import sklearn
import pickle
import sys
sys.path.append("../../src/llms/mistral-origin")  # append the path where mistral-src was cloned

from tqdm import tqdm
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from matplotlib.cm import rainbow
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer

def load_model(model_path):
    """
    Load the model and tokenizer from ckpts 
    args:
        model_path: the path of ckpt
    """
    model_path = Path(model_path)
    model = Transformer.from_folder(model_path, dtype=torch.bfloat16)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    print(f"Successfully load the model and tokenizer from mistral in {model_path}.")
    return model, tokenizer

def load_data(data_path, name="all"):
    """
    Load data
    args:
        data_path: data path in .csv format
    """
    if not os.path.exists(data_path):
        print(f"The path {data_path} does not exists.")
        return False
    
    data = [] # list of (text, label)
    data_class = []
    data_path = Path(data_path)
    with open(data_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:  # skip csv header
                continue
            data.append((row[0], row[1]))
            data_class.append(int(row[1]))

    # label ID
    labels = sorted({x[1] for x in data}) # ["1","2","3","4"]
    print(f"Reloaded {name} data {len(data)} samples with {len(labels)} labels.")

    return data, data_class

def emb_data(data, model, tokenizer):
    """
    Convert the text info to vector format by using embedding methods.
    args:
        data: input data for embedding
        model: input ckpts of mistral
        tokenizer: for encoding
    returns:
        X: data after embedding
    """
    print(f"Successfully deploy the model to gpu device {model.device}.")
    with torch.no_grad():
        featurized_x = []
        # compute an embedding for each sentence
        for i, (x, y) in tqdm(enumerate(data)):
            tokens = tokenizer.encode(x, bos=True)
            tensor = torch.tensor(tokens).to(model.device)
            features = model.forward_partial(tensor, [len(tokens)])  # (n_tokens, model_dim)
            featurized_x.append(features.float().mean(0).cpu().detach().numpy())

    # concatenate sentence embeddings
    X = np.concatenate([x[None] for x in featurized_x], axis=0)  # (n_points, model_dim)
    return X

def split(X, data_class, train_test_scale=0.8):
    """
    split the data into train and test set
    args:
        X: input embeddings
        data_class: labels
    returns:
        train_x, train_y, test_x, test_y
    """
    rng = np.random.default_rng(seed=2024)
    # seed = 2401: 77%
    # seed = 2024: 77.5%

    # shuffle
    permuted = rng.permutation(len(X))
    shuffled_x, shuffled_y = X[permuted], np.array(data_class)[permuted]

    # train/test split
    train_num = int(len(shuffled_x) * train_test_scale)
    train_x, train_y = shuffled_x[:train_num], shuffled_y[:train_num]
    test_x, test_y = shuffled_x[train_num:], shuffled_y[train_num:]

    # summary
    print(f"Train set : {len(train_x)} samples")
    print(f"Test set  : {len(test_x)} samples")
    return train_x, train_y, test_x, test_y

def normalize(x):
    """
    Normalize the input data
    args:
        x: input features after embedding
    returns:
        norm_x: after normalization
    """
    scaler = StandardScaler()
    norm_x = scaler.fit_transform(x)
    return norm_x

def LR(train_x, train_y, test_x, test_y):
    """
    Classification model: Logistic Regression
    args:
        train_x, train_y: train data and train label
        test_x, test_y: test data and test label
    """
    lr = LogisticRegression(random_state=42, C=1.0, max_iter=300).fit(train_x, train_y)
    acc = np.mean(lr.predict(test_x) == test_y)
    print(f"Accuracy: {100*acc: .2f}%")
    return lr, acc

def vis_tsne(X, data_class, save_img="tsne.jpg"):
    """
    Visualization the T-SNE Distribution
    args: 
        x: data for visualize
    """
    reduced = sklearn.manifold.TSNE(n_components=2, random_state=0).fit_transform(X)
    plt.scatter(reduced[:,0], reduced[:,1], c = data_class, cmap="rainbow")
    plt.savefig(save_img)
    # plt.show()

def lr_train(pretrained_path, data_path, tsne_name, save_path):
    """
    Finish the LR train process, including the data preparation.
    args:
        pretrained_path: path of llm pretrained model from ckpts
        data_path: path of all text data in csv format
        tsne_name: save t-sne distribution jpg
        save_path: path for save text classification model in .pickle
    """
    # Load ckpt and data
    pretrained_model, tokenizer = load_model(pretrained_path) # 77%
    data, data_class =  load_data(data_path, name="all")

    # emb
    embeddings = emb_data(data, pretrained_model, tokenizer)

    # t-sne visualisation
    vis_tsne(embeddings, data_class, save_img=tsne_name)

    # train
    train_x, train_y, test_x, test_y = split(embeddings, data_class, train_test_scale=0.8)
    # train_x = normalize(train_x)
    # test_x = normalize(test_x)
    lr, acc = LR(train_x, train_y, test_x, test_y)

    # save
    with open(save_path, 'wb') as file:
        pickle.dump(lr, file)
    print(f"Successuflly save the text classification model into {save_path}")

def lr_predict(model, pred_x, save_csv):
    """
    predict the zero.csv file, save to zero_predict.csv
    args:
        model: trained classification model
        pred_x: for zero.csv
        save_csv: path for zero_predict.csv
    """
    pass

if __name__ == "__main__":
    # pretrained_path = "../../ckpts/Mistral-7B-Instruct-v0.2-origin/" # 74%
    pretrained_path = "../../ckpts/mistral-7B-v0.1/" # 77%
    data_path = "../data/non_zero.csv"
    tsne_name = "tsne.jpg"

    model_save_path = "./ckpts/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model_name = "LR_" + pretrained_path.split("/")[-2] + ".pickle"
    save_path = os.path.join(model_save_path, model_name)
    print(f"save path: {save_path}")

    # LR train and model save
    lr_train(pretrained_path, data_path, tsne_name, save_path)

    # LR predict
    pred_x = "./data/zero.csv"
    # data to emb
    pretrained_model, tokenizer = load_model(pretrained_path) # 77%
    pred_data, data_class =  load_data(data_path, name="predict")
    pred_embeddings = emb_data(predict_data, pretrained_model, tokenizer)
    # lr_predict(save_path, pred_x)
