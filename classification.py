import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel

# Load and preprocess the data
df_train = pd.read_csv('C:/Users/ashen/Downloads/requesition/train/df.csv')
df_valid = pd.read_csv('C:/Users/ashen/Downloads/requesition/validation/df.csv')

# Tokenize the input sequences using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_sequences = tokenizer(df_train['tokens'].tolist(), padding=True, truncation=True, max_length=128)
valid_sequences = tokenizer(df_valid['tokens'].tolist(), padding=True, truncation=True, max_length=128)

# Prepare the labels
train_labels = df_train['y'].values
valid_labels = df_valid['y'].values

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(train_sequences['input_ids'], train_labels, test_size=0.2, random_state=42)

# Create input tensors for BERT model
input_ids = Input(shape=(128,), dtype=tf.int32)
attention_mask = Input(shape=(128,), dtype=tf.int32)

# Load the BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_output = bert_model(input_ids)[0]

# Add additional layers on top of BERT
dropout_rate = 0.2
dense_units = 128
output_units = 1  # Change the output units to 1 for binary classification

x = Dropout(dropout_rate)(bert_output[:, 0, :])  # Use only the first token embedding
x = Dense(dense_units, activation='relu')(x)
x = Dropout(dropout_rate)(x)
output = Dense(output_units, activation='sigmoid')(x)

# Create the model
model = Model(inputs=[input_ids, attention_mask], outputs=output)

# Compile the model
optimizer = Adam(learning_rate=2e-5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

# Train the model
epochs = 10
batch_size = 32
model.fit([np.array(train_X), np.ones_like(train_X)], np.array(train_y), validation_data=([np.array(val_X), np.ones_like(val_X)], np.array(val_y)), epochs=epochs, batch_size=batch_size)

# Evaluate the model on the validation set
val_predictions = model.predict([np.array(val_X), np.ones_like(val_X)])
val_predictions = (val_predictions > 0.5).astype(int)  # Convert probabilities to binary predictions
val_f1 = f1_score(val_y, val_predictions)
print("Validation F1 Score:", val_f1)

# Save the model
model.save('model.h5')






