"""
Evrything you need to train a text generation model

The model is a simple RNN with an embedding layer, a GRU layer, and a fully connected layer.

Example:

    >>> from model import *
    >>> from data import *
    >>> import torch
    >>> model = RNN(len(vocab), 1024, 1024, 2)
    >>> data = DataLoader(Wikipedia(50))
    >>> optim = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> criterion = nn.CrossEntropyLoss()
    >>> for epoch in range(10): 
    >>>     loss, acc = train(model, data, optim, criterion, accuracy_fn=accuracy, save_path='model.pth')
    >>>     print(f'Epoch: {epoch+1}/{N_EPOCHS}: Loss = {loss_}, Accuracy = {acc_}')
    """

import torch
import torch.nn as nn
from data import vocab
import tqdm

# Define the RNN model
class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) model for language modeling.
    
    This class implements a GRU-based RNN for generating text sequences. It supports 
    optional embedding weight sharing and includes a method for generating random 
    sentences using temperature-based sampling.
    
    Attributes:
        embedding (nn.Embedding): Embedding layer for input tokens
        rnn (nn.GRU): Gated Recurrent Unit (GRU) layer for sequence processing
        fc (nn.Linear): Fully connected layer for output prediction
    
    Methods:
        forward: Processes input through embedding and RNN layers
        random_sent: Generates a random sentence using the trained model
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, use_emmbedding_as_fc = True):
        """
        Initialize the RNN model with configurable embedding, RNN, and fully connected layers.
        
        Args:
            vocab_size (int): Size of the vocabulary for embedding and output layers
            embedding_dim (int): Dimensionality of the token embeddings
            hidden_dim (int): Number of features in the hidden state of the RNN
            n_layers (int): Number of GRU layers in the RNN
            use_emmbedding_as_fc (bool, optional): Whether to share embedding weights with output layer. Defaults to True.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        if use_emmbedding_as_fc:
            self.fc.weight.data = self.embedding.weight.data
    
    def forward(self, x) -> torch.Tensor:
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)
    
    def random_sent(self, max_length, t = 1., initial_sent = ['<SOS>']) -> str:
        """
        Generate a random sentence using the trained RNN model with temperature-based sampling.
    
        Args:
            max_length (int): Maximum length of the generated sentence.
            t (float, optional): Temperature parameter for controlling randomness. Defaults to 1.0.
            initial_sent (list, optional): Initial seed words for sentence generation. Defaults to ['<SOS>'].
        
        Returns:
            str: Generated sentence with special tokens removed and words joined by spaces.
        """
        sent = initial_sent
        x = torch.tensor([[vocab.index(word) for word in sent]])
        x = self.embedding(x)
        output, (h, c) = self.rnn(x)
        output = self.fc(output)
        prob = torch.softmax(output[0, -1] / t, dim=-1)
        word_idx = torch.multinomial(prob, 1)
        sent.append(vocab[word_idx])
        for _ in range(max_length - len(sent)):
            x = torch.tensor([[vocab.index(sent[-1])]])
            x = self.embedding(x)
            output, (h, c) = self.rnn(x, (h, c))
            output = self.fc(output)
            prob = torch.softmax(output[0, -1] / t, dim=-1)
            word_idx = torch.multinomial(prob, 1)
            sent.append(vocab[word_idx])
        setn_text = ''
        for i in range(len(sent)):
            if sent[i] == '<SOS>':
                continue
            elif sent[i] == '<EOS>':
                break
            elif sent[i] == '<CAT>':
                if i != 0:
                    del setn_text[-1]
            else:
                setn_text += sent[i] + ' '
        return setn_text

# TODO: add more models like GPT

# Define the training function
def train(model, iterator, optimizer, criterion, accuracy_fn = None, save_path = None) -> tuple[float, float] | tuple[float, None]:
    """
    Train a neural network model for a single epoch.
    
    Args:
        model (nn.Module): The neural network model to train.
        iterator (DataLoader): Data iterator containing training batches. It should yield batches of input data (x) and target labels (y).
        optimizer (torch.optim.Optimizer): Optimization algorithm for updating model parameters.
        criterion (nn.Module): Loss function for computing model error.
        accuracy_fn (callable, optional): Function to compute model accuracy. Defaults to None.
        save_path (str, optional): Path to save model state dictionary. Defaults to None.
    
    Returns:
        tuple: A tuple containing:
            - Average loss for the epoch (float)
            - Average accuracy for the epoch (float or None)
    """
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for x, y in tqdm.tqdm(iterator):
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions.reshape(-1, len(vocab)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if accuracy_fn:
            acc = accuracy_fn(predictions, y)
            epoch_acc += acc
    if save_path:
        torch.save(model.state_dict(), save_path)
    if accuracy_fn:
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    else:
        return epoch_loss / len(iterator), None
    
# Define the accuacy function
def accuracy(yp, yt) -> float:
    """
    Calculate the accuracy of predictions, adjusting for padding.
    
    Args:
        yp (torch.Tensor): Predicted labels with shape (batch_size, num_classes)
        yt (torch.Tensor): True labels with shape (batch_size)
    
    Returns:
        float: Accuracy score normalized by removing padding tokens
    """
    acc = (yp.argmax(-1) == yt).float().mean().item()
    pad = (yt == vocab.index("<PAD>")).float().mean().item()
    return acc / (1 - pad)

# Define the evaluation function
def evaluate(model, iterator, criterion, accuracy_fn = None) ->  tuple[float, float] | tuple[float, None]:
    """
    Evaluate a neural network model on a given dataset.
    
    Args:
        model (nn.Module): The neural network model to evaluate.
        iterator (DataLoader): Data iterator containing evaluation batches. It should yield batches of input data (x) and target labels (y).
        criterion (nn.Module): Loss function for computing model error.
        accuracy_fn (callable, optional): Function to compute model accuracy. Defaults to None.
    
    Returns:
        tuple: A tuple containing:
            - Average loss for the evaluation (float)
            - Average accuracy for the evaluation (float or None)
    """
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for x, y in iterator:
            predictions = model(x).squeeze(1)
            loss = criterion(predictions, y)
            epoch_loss += loss.item()
            if accuracy_fn:
                acc = accuracy_fn(predictions, y)
                epoch_acc += acc
    if accuracy_fn:
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    else:
        return epoch_loss / len(iterator), None