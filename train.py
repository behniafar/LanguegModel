import torch
import torch.nn as nn
from torch import optim
from data import *
from model import *
import os

# Define the hyperparameters
INPUT_DIM = len(vocab)
EMBEDDING_DIM = 1000
HIDDEN_DIM = 1000
N_LAYERS = 2
LEARNING_RATE = 0.001
N_EPOCHS = 100
BATCH_SIZE = 100
MAX_LENGTH = 50
ROOT = 'C://Users/mahdi/Desktop/myProjects/python/NovelGenerator/'
SAVE_PATH = ROOT + 'models/NG_E' + str(EMBEDDING_DIM) + '_H' + str(HIDDEN_DIM) + '.pth'

# Define the model
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.index('<PAD>'))

# Load data
train_iterator = DataLoader(wikipedia(max_length=MAX_LENGTH, save_path=ROOT + 'data/wiki' + str(MAX_LENGTH) + '.pt'), batch_size=BATCH_SIZE, shuffle=True)
print('data loaded')

# Load the saved model if exists
if os.path.exists(SAVE_PATH):
    try:
        model.load_state_dict(torch.load(SAVE_PATH))
        print('Model loaded successfully!')
    except:
        print('Model not found. Training from scratch.')
else:
    print('Model not found. Training from scratch.')

if __name__ == '__main__':
    # Train
    acc = []
    loss = []
    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch+1}/{N_EPOCHS}:')
        loss_, acc_ = train(model, train_iterator, optimizer, criterion, accuracy_fn=accuracy, save_path=SAVE_PATH)
        loss.append(loss_)
        acc.append(acc_)
        print(f'Loss: {loss_}, Accuracy: {acc_}')
        print()
