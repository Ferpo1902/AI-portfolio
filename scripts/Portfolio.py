#implementing portfolio strategies
import lightgbm as lgb
import pandas as pd
from FeatureEngine import *
import os
import pickle
from clean_analysis import Model

def retrieve_model(model_folder):
    """gets model information from model folder

    Args:
        model_folder (str): path to the model folder

    Raises:
        Exception: feature cannot be found

    Returns:
        tuple: info for the model
    """
    lgb_model = lgb.Booster(model_file = f"{model_folder}/model.txt")
    info = pd.read_csv(f"{model_folder}/info.csv")
    dates = pd.read_csv(f"{model_folder}/dates.csv")
    features = []
    for feature_file in os.listdir(f"{model_folder}/features"):
        try:
            with open(f"{model_folder}/features/{feature_file}", 'rb') as file:
                # Load the object from the file
                feature = pickle.load(file)
        except:
            raise Exception(f"Feature {feature_file} could not be read")
        features.append(feature)
    with open(f"{model_folder}/target.pkl", 'rb') as file:
        target = pickle.load(file)
    return lgb_model, info, dates, features, target

class Portfolio:
    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.get_model()

    def get_model(self):
        lgb_model, info, dates, features, target = retrieve_model(self.model_folder)
        self.model = Model(universe_path = info.universe_path[0])
        self.model.add_features(features)
        self.model.add_target(target)
        self.model.model = lgb_model
        self.model.data = self.model.data[(self.model.data.date >= dates.test_start) & 
                                          (self.model.data.date >= dates.test_end)]
        self.model.generate_targets_and_features()

    

