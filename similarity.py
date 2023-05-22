import os
import subprocess
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("loading standard imports")
import os
import sys
import json
import shutil
from argparse import ArgumentParser
import traceback

src = 'C:/Users/ashen/Downloads/requesition/'

try:
    from src.preprocessing import tokenize_function, encode_function
except ImportError:
    from preprocessing import tokenize_function, encode_function
except Exception as e:
    traceback.print_exc()
    raise e


def main(args):
    embeddings_df = pd.read_csv('C:/Users/ashen/Downloads/requesition/train/embeddings.csv')
    embeddings_df['embeddings'] = embeddings_df['embeddings'].apply(lambda x: np.array(json.loads(x)))

    def get_cosine_similarity(input_text):
        embedding = encode_function(input_text)[np.newaxis]
        print(f"Getting cosine similarity scores on {embedding.shape}")
        score = cosine_similarity(embedding, np.stack(embeddings_df['embeddings'].values))
        embeddings_df['Similarity'] = score[0]
        embeddings_df.sort_values(by='Similarity', ascending=False, inplace=True)
        top_results_indices = embeddings_df.index[:10]
        return top_results_indices

    # Example usage
    input_text = "TK61504643T ECM Manual Motor Control, PWM Signal Output Type, 24 V AC Input Voltage, 10 mA Maximum Current"
    top_results_indices = get_cosine_similarity(input_text)
    print("Print the top results")
    for idx in top_results_indices:
        print("Material Description:", embeddings_df.loc[idx, 'material_description'])
        print("Supplier Part Number:", embeddings_df.loc[idx, 'supplier_part_number'])
        print("Supplier ID:", embeddings_df.loc[idx, 'vendor_number'])
        print("Supplier Name:", embeddings_df.loc[idx, 'supplier_name_l1'])
        print("Requisition Date:", embeddings_df.loc[idx, 'requisition_date'])
        print("Similarity:", embeddings_df.loc[idx, 'Similarity'])
        print()


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
