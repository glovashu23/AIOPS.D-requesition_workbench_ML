import subprocess
import os
import json
from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from multiprocessing import Pool
from tqdm.auto import tqdm
import multiprocessing
from sentence_transformers import SentenceTransformer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
BASE_DIR = "C:\\Users\\ashen\\Downloads\\requesition"
output_dir = os.path.join(BASE_DIR, 'train')

def encode_function(text):
    encoding = model.encode(text)
    return encoding

def tokenize_function(text):
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    MAX_LEN = 128
    tokenized = pad_sequences([tokenized], maxlen=MAX_LEN, truncating="post", padding="post")
    return tokenized[0]


model = SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')


def main(args):
    catalog_data_df = pd.read_csv(os.path.join(BASE_DIR, 'catalog_data.csv'), sep='|')
    catalog_data_df = catalog_data_df.drop_duplicates(subset=['material_description'], keep='last').dropna().reset_index(drop=True)
    catalog_data_df['line_type'] = 'Catalog Item'
    catalog_data_df['y'] = 0
    series = catalog_data_df['material_description']
    catalog_data_df["embeddings"] = [json.dumps(encode_function(x).tolist()) for x in tqdm(series)]
    catalog_data_df["tokens"] = [json.dumps(tokenize_function(x).tolist()) for x in tqdm(series)]

    noncat_data_df = pd.read_csv(os.path.join(BASE_DIR, 'noncat_data.csv'), sep='|')
    noncat_data_df = noncat_data_df.drop_duplicates(subset=['material_description'], keep='last').dropna().reset_index(drop=True)
    noncat_data_df['line_type'] = 'Non-Catalog Item'
    noncat_data_df['y'] = 1
    series = noncat_data_df['material_description']
    noncat_data_df["embeddings"] = [json.dumps(encode_function(x).tolist()) for x in tqdm(series)]
    noncat_data_df["tokens"] = [json.dumps(tokenize_function(x).tolist()) for x in tqdm(series)]

    noncat_train, noncat_testvalid = train_test_split(noncat_data_df, test_size=args.valid_split + args.test_split, random_state=args.seed)
    noncat_valid, noncat_test = train_test_split(noncat_testvalid, test_size=args.test_split / (args.valid_split + args.test_split), random_state=args.seed)

    catalog_train, catalog_testvalid = train_test_split(catalog_data_df, train_size=len(noncat_train) / (len(catalog_data_df) * args.balance), random_state=args.seed)
    catalog_valid, catalog_test = train_test_split(catalog_testvalid, test_size=args.test_split / (args.valid_split + args.test_split), random_state=args.seed)

    df_train = pd.concat([noncat_train, catalog_train])
    df_valid = pd.concat([noncat_valid, catalog_valid])
    df_test = pd.concat([noncat_test, catalog_test])

    os.makedirs(os.path.join(BASE_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'test'), exist_ok=True)
    
    df_train.sample(frac=1).to_csv(os.path.join(BASE_DIR, 'train', 'df.csv'), header=True, index=False)
    catalog_data_df.to_csv(os.path.join(BASE_DIR, 'train', 'embeddings.csv'), header=True, index=False)
    df_valid.to_csv(os.path.join(BASE_DIR, 'validation', 'df.csv'), header=True, index=False)
    df_test.to_csv(os.path.join(BASE_DIR, 'test', 'df.csv'), header=True, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--balance', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
