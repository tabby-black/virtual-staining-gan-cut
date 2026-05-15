# jupyter notebook transferred to python file for submission

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import regionprops_table


# load data from a csv file and plot histogram 
def read_csv_and_plot_histogram(csv_path, feature):
    df = pd.read_csv(csv_path)
    # nuclei counts is restored to an array from the csv
    counts = df["nuclei_count"].values
    mean_areas = df["mean_areas"].values
    mean_eccentricities = df["mean_eccentricities"].values

    if feature == "counts":

        plt.figure(figsize=(6, 4))
        sns.histplot(counts, bins=10, kde=True)

        plt.xlabel("Nuclei count per image")
        plt.ylabel("Frequency")
        plt.title("Nuclei count frequency distribution")

        plt.xlim(left=0)

        plt.show()
        
    elif feature == "areas":

        plt.figure(figsize=(6, 4))
        sns.histplot(mean_areas, bins=12, kde=True)

        plt.xlabel("Mean nuclei area per image")
        plt.ylabel("Frequency")
        plt.title("Mean nuclei area frequency distribution")

        plt.xlim(left=0)

        plt.show()
        
    else:
        plt.figure(figsize=(6, 4))
        sns.histplot(mean_eccentricities, bins=10, kde=True)

        plt.xlabel("Mean nuclei eccentricity per image")
        plt.ylabel("Frequency")
        plt.title("Mean nuclei eccentricity frequency distribution")

        plt.xlim(left=0)

        plt.show()


# read two csv files and overlay the histograms on top of each other

def read_csv_and_plot_joint_histogram(csv_path_fake, feature, csv_path_real="../results/histology_analysis/real_RGB_nuclei_features.csv"):
    
    df_real = pd.read_csv(csv_path_real)
    df_fake = pd.read_csv(csv_path_fake)
    # nuclei counts is restored to an array from the csv
    counts_real = df_real["nuclei_count"].values
    mean_areas_real = df_real["mean_areas"].values
    mean_eccentricities_real = df_real["mean_eccentricities"].values

    counts_fake = df_fake["nuclei_count"].values
    mean_areas_fake = df_fake["mean_areas"].values
    mean_eccentricities_fake = df_fake["mean_eccentricities"].values

    if feature == "counts":

        plt.figure(figsize=(6, 4))
        sns.histplot(counts_real, bins=10, kde=True, label="Real RGB")
        sns.histplot(counts_fake, bins=10, kde=True, label="Generated RGB")

        plt.xlabel("Nuclei count per image")
        plt.ylabel("Frequency")
        plt.title("Nuclei count frequency distribution")

        plt.xlim(left=0)

        plt.show()
        
    elif feature == "areas":

        plt.figure(figsize=(6, 4))
        sns.histplot(mean_areas_real, bins=10, kde=True, label="Real RGB")
        sns.histplot(mean_areas_fake, bins=10, kde=True, label="Generated RGB")
        
        plt.xlabel("Mean nuclei area per image")
        plt.ylabel("Frequency")
        plt.title("Mean nuclei area frequency distribution")

        plt.xlim(left=0)

        plt.show()
        
    else:
        
        plt.figure(figsize=(6, 4))
        sns.histplot(mean_eccentricities_real, bins=10, kde=True, label="Real RGB")
        sns.histplot(mean_eccentricities_fake, bins=10, kde=True, label="Generated RGB")

        plt.xlabel("Mean nuclei eccentricity per image")
        plt.ylabel("Frequency")
        plt.title("Mean nuclei eccentricity frequency distribution")

        plt.xlim(left=0)

        plt.show()


# read multiple csv files and overlay the kdes on top of each other
def read_csv_and_plot_joint_kde(csv_paths_fake, feature, csv_path_real="../results/histology_analysis/real_RGB_nuclei_features.csv"):
    
    df_real = pd.read_csv(csv_path_real)
    # nuclei counts is restored to an array from the csv
    counts_real = df_real["nuclei_count"].values
    mean_areas_real = df_real["mean_areas"].values 
    mean_eccentricities_real = df_real["mean_eccentricities"].values


    if feature == "counts":
        
        plt.figure(figsize=(6, 4))
        sns.kdeplot(counts_real, linewidth=3, label="Real RGB")

        for run_name, csv_path_fake in csv_paths_fake.items():
            df_fake = pd.read_csv(csv_path_fake)
            counts_fake = df_fake["nuclei_count"].values
            sns.kdeplot(counts_fake, linewidth=1.5, label=run_name)

        plt.xlabel("Nuclei count per image")
        plt.ylabel("Frequency")
        plt.title("Nuclei count frequency distribution")

        plt.xlim(left=0)

        plt.show()
        
    elif feature == "areas":

        plt.figure(figsize=(6, 4))
        sns.kdeplot(mean_areas_real, linewidth=3, label="Real RGB")

        for run_name, csv_path_fake in csv_paths_fake.items():
            df_fake = pd.read_csv(csv_path_fake)
            mean_areas_fake = df_fake["mean_areas"].values
            sns.kdeplot(mean_areas_fake, linewidth=1.5, label=run_name)
        
        plt.xlabel("Mean nuclei area per image")
        plt.ylabel("Frequency")
        plt.title("Mean nuclei area frequency distribution")

        plt.xlim(left=0)

        plt.show()
        
    else:
        
        plt.figure(figsize=(6, 4))
        sns.kdeplot(mean_eccentricities_real, linewidth=3, label="Real RGB")

        for run_name, csv_path_fake in csv_paths_fake.items():
            df_fake = pd.read_csv(csv_path_fake)
            mean_eccentricities_fake = df_fake["mean_eccentricities"].values
            sns.kdeplot(mean_eccentricities_fake, linewidth=1.5, label=run_name)

        plt.xlabel("Mean nuclei eccentricity per image")
        plt.ylabel("Frequency")
        plt.title("Mean nuclei eccentricity frequency distribution")

        plt.xlim(left=0)

        plt.show()


csv_paths_fake = {
    "CycleGAN rep1": "../results/histology_analysis/rep1_e5_fake_RGB_nuclei_features.csv",
    "CycleGAN rep1a": "../results/histology_analysis/rep1a_e170_fake_RGB_nuclei_features.csv",
    "CycleGAN rep2": "../results/histology_analysis/rep2_e190_fake_RGB_nuclei_features.csv",
    "CycleGAN rep2a": "../results/histology_analysis/rep2a_e200_fake_RGB_nuclei_features.csv",
}

read_csv_and_plot_joint_kde(csv_paths_fake, "counts")
read_csv_and_plot_joint_kde(csv_paths_fake, "areas")
read_csv_and_plot_joint_kde(csv_paths_fake, "eccentricities")