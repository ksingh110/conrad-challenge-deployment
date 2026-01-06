import pandas as pd
import pickle
import os

FEATURE_NAMES_PATH = (
    "/Users/krishaysingh/Documents/Holland_Lab_KidneyCancer/"
    "fred-hutch-immunotherapy-code/lfc_5/feature_names_lfc5.pkl"
)

DATASET_PATH = "/Users/krishaysingh/Downloads/dataset_head_and_neck.csv"
OUTPUT_DIR = "/Users/krishaysingh/Downloads/test_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

df = df.set_index(df.columns[1])

df = df.T

feature_names = pickle.load(open(FEATURE_NAMES_PATH, "rb"))

X = df.loc[:, feature_names]

sampled_rows = X.sample(n=50, random_state=None)

for sample_name, row in sampled_rows.iterrows():
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{sample_name}_testData.csv"
    )
    row.to_frame().T.to_csv(output_path, index=False)

print(OUTPUT_DIR)
