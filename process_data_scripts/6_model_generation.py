import pandas as pd
import numpy as np
#import openpyxl
import os
from tqdm import tqdm
tqdm.pandas()
import csv

import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf
import random
import matplotlib.pyplot as plt

def divide_games_by_id(df):
    df.index = df.index.str.split('-').str[0]

    games = df.index.unique()

    games = games.tolist()
    train_ids = random.sample(games, k=round(len(games) * 0.7))
    games = list(set(games) - set(train_ids))
    validation_ids = random.sample(games, k=round(len(games) * 0.5))
    test_ids = list(set(games) - set(validation_ids))

    train_df = df.loc[train_ids]
    validation_df = df.loc[validation_ids]
    test_df = df.loc[test_ids]

    return train_df, validation_df, test_df

def main():

    random.seed(1234)
    tf.random.set_seed(1234)

    #tf.config.threading.set_intra_op_parallelism_threads(1)
    #tf.config.threading.set_inter_op_parallelism_threads(12)

    BASE_FOLDER = '../data/5_lichess_csv_ML_ready'
    OUTPUT_FOLDER= '../data/6_models'
    OUTPUT_FIGS = '../data/6_models/figs'
    OUTPUT_TUNING = '../data/6_models/tuning'
    OUTPUT_ACCURACY = "../data/Model_accuracy.csv"

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    if not os.path.exists(OUTPUT_FIGS):
        os.makedirs(OUTPUT_FIGS)
    
    if not os.path.exists(OUTPUT_TUNING):
        os.makedirs(OUTPUT_TUNING)
    
    print(os.listdir(BASE_FOLDER))

    model_accuracies = {}

    for csv_file_name in tqdm(os.listdir(BASE_FOLDER)):
        print(f"Reading file {csv_file_name}")

        if not csv_file_name.endswith('.pkl'):
            continue

        #ignore all files between 1200 and 1900
        #print(csv_file_name.split('_')[1])
        #if int(csv_file_name.split('_')[1]) <= 1900 and int(csv_file_name.split('_')[1]) >= 1200:
        #    continue

        if os.path.exists(f'{OUTPUT_FOLDER}/{csv_file_name}'):
            print(f"Model already exists for {csv_file_name}")
            continue

        try:
        
            INPUT_FILE_PATH = f'{BASE_FOLDER}/{csv_file_name}'
            OUTPUT_FILE = f'{OUTPUT_FOLDER}/{csv_file_name}'
        
            df = pd.read_pickle(INPUT_FILE_PATH)

            #! remove draws for now
            df = df[df['Result'] != 2]

            #! remove unwanted columns in the training
            df = df.drop(columns=['Board','MoveNumber','WhiteElo','BlackElo'])
            #df = df.reset_index(drop=True)
            #df = df.reindex(sorted(df.columns), axis=1)

            # Load the training, validation, test sets (70% train, 15% validation, 15% test)
            #train_df, test_df = train_test_split(df, test_size=0.3, random_state=1234)
            #validation_df, test_df = train_test_split(test_df, test_size=0.5, random_state=1234)

            # Split the data by game id, so that the same game is not in both training and test
            # This is to avoid data leakage
            # (70% train, 15% validation, 15% test)
            train_df, validation_df, test_df = divide_games_by_id(df)
            # just to make sure the columns are in the same order as the ones in the bot
            train_df = df.reset_index(drop=True)
            validation_df = df.reset_index(drop=True)
            test_df = df.reset_index(drop=True)
            train_df = train_df.reindex(sorted(train_df.columns), axis=1)
            validation_df = validation_df.reindex(sorted(validation_df.columns), axis=1)
            test_df = test_df.reindex(sorted(test_df.columns), axis=1)

            train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="Result")
            validation_ds = tfdf.keras.pd_dataframe_to_tf_dataset(validation_df, label="Result")
            test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="Result")

            # Create a Random Search tuner with 10 trials and automatic hp configuration.
            tuner = tfdf.tuner.RandomSearch(num_trials=10, use_predefined_hps=True, trial_maximum_training_duration_seconds=3600)

            # Train the model
            model = tfdf.keras.RandomForestModel(random_seed=1234,tuner=tuner,verbose=2)
            #model = tfdf.keras.RandomForestModel(random_seed=1234)
            model.compile(metrics=["accuracy"])
            model.fit(train_ds, validation_data=validation_ds,verbose=2)

            # Evaluate the model
            accuracy = model.evaluate(test_ds)[1]
            print(accuracy)
            model_accuracies[csv_file_name] = accuracy

            model.save(OUTPUT_FILE)
 
            # Save the tuning logs
            tuning_logs = model.make_inspector().tuning_logs()
            tuning_logs[tuning_logs.best].iloc[0]
            tuning_logs[tuning_logs.best].iloc[0].to_csv(f'{OUTPUT_TUNING}/{csv_file_name}_tuning_param.csv')

            # Plot the tuning score, between the best trial and another trial
            plt.figure(figsize=(10, 5))
            plt.plot(tuning_logs["score"], label="current trial")
            plt.plot(tuning_logs["score"].cummax(), label="best trial")
            plt.xlabel("Tuning step")
            plt.ylabel("Tuning score")
            plt.legend()
            plt.savefig(f'{OUTPUT_FIGS}/{csv_file_name}_tuning_score.png')


        except Exception as e:
            print(f"Error processing file {csv_file_name}")
            print(e)
            continue
    
    with open(OUTPUT_ACCURACY, 'w') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = ['Model', 'Accuracy']) 
        writer.writeheader() 
        for i in model_accuracies:
            writer.writerow({'Model': i, 'Accuracy': model_accuracies[i]})

if __name__ == "__main__":
    main()