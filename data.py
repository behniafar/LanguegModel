"""
Collection text data

You can use random text from the web or wikipedia like this:

    >>> from data import *
    >>> data = DataLoader(Wikipedia(50))

Or a novel from fanmtl.com like this:

    >>> data = DataLoader(FanmtlNovel('shado_slave'))
"""

import re
import os
import requests
import time
from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import tqdm
import nltk
import random
import exceptions
nltk.download('punkt_tab')

with open('data/vocab.txt', 'r', encoding='utf-8') as f:
    vocab = [word[:-1] for word in f.readlines()] # [:-1] to remove '\n'

def filter(text : str):
    # Change capital letters to lowercase
    text = text.lower()

    # Delete unknown characters
    allowed_chars = set('abcdefghijklmnopqrstuvwxyz0123456789.,!?;:"\'()[]{}*-…/—% ')
    text = ''.join([char for char in text if char in allowed_chars])
    return text

def process(text : list[str]) -> list[str]:
    """
    Process and normalize text data for natural language processing tasks.
    
    This function performs several text preprocessing steps:
    - Lowercase conversion
    - Character filtering
    - Tokenization and word normalization
    - Vocabulary building with special tokens
    - Word splitting and exception handling
    
    Args:
        text (list[str]): A list of text strings to be processed
    
    Returns:
        tuple: A tuple containing:
            - Processed text as a list of tokenized sentences
            - A vocabulary set with special tokens
    """
    # Change capital letters to lowercase
    text = [line.lower() for line in text]

    # Delete unknown characters
    allowed_chars = set('abcdefghijklmnopqrstuvwxyz0123456789.,!?;:"\'()[]{}*-…/—% ')
    text = [''.join([char for char in line if char in allowed_chars]) for line in text]

    # Put spaces between words and symbols
    # For example, "it's a test." becomes "it ' s a test ."
    text = [' '.join(re.findall(r'\w+|\S', line)) for line in text]

    # Replace numbers with '<NUM>'
    text = [re.sub(r'\d+', ' <NUM> ', line) for line in text]

    # Build a vocab
    vocab = set(' '.join(text).split())

    # Prefixes and suffixe
    fixes = ['re', 'over', 'ing', 'ed', 'y', 'a', 'ly', 'able', 'less', 'ful', 'ness', 'er', 'est', 'ity', 'hood', 'ship', 'ment', 'dis', 'un']

    # Add suffixes and prefixes
    for i in fixes: vocab.add(i)

    # Remove invalid words
    with open('data/invalid_tokens.txt', 'r', encoding='utf-8') as f:
        invalid_tokens = [line[:-1] for line in f.readlines() if line != '\n']
    for i in invalid_tokens:
        vocab.discard(i)

    # Split sentences into words
    text = [line.split() for line in text]

    # Split 's'
    for i in range(len(text)):
        for j in range(len(text[i])):
            added = 0
            if len(text[i][j + added]) > 3:
                if text[i][j + added].endswith('s') and text[i][j][:-1] in vocab:
                    text[i][j + added] = text[i][j + added][:-1]
                    text[i].insert(j+1 + added, '<CAT>')
                    text[i].insert(j+2 + added, 's')
                    added += 2


    # Remove added characters before 'er' and 'est' and 'ed' and 'ing'
    # for example, 'bigger' becomes 'biger' because 'big' is in vocab an 'g' is added before 'er'
    # <CAT> is a special token that means 'concatenate'
    for i in range(len(text)):
        for j in range(len(text[i])):
            if len(text[i][j]) > 5:
                if text[i][j].endswith('er'):
                    if text[i][j][-3] == text[i][j][-4] and  text[i][j][:-3] in vocab:
                        text[i].insert(j+1, '<CAT> er')
                        text[i][j] = text[i][j][:-3]
                elif text[i][j].endswith('ed'):
                    if text[i][j][-3] == text[i][j][-4] and  text[i][j][:-3] in vocab:
                        text[i].insert(j+1, '<CAT> ed')
                        text[i][j] = text[i][j][:-3]
                elif text[i][j].endswith('est'):
                    if text[i][j][-4] == text[i][j][-5] and  text[i][j][:-4] in vocab:
                        text[i].insert(j+1, '<CAT> est')
                        text[i][j] = text[i][j][:-4]
                elif text[i][j].endswith('ing'):
                    if text[i][j][-4] == text[i][j][-5] and  text[i][j][:-4] in vocab:
                        text[i].insert(j+1, '<CAT> ing')
                        text[i][j] = text[i][j][:-4]
                        
    # Remove characters befor splitting
    discarded = []
    for i in 'abcdefghijklmnopqrstuvwxyz':
        vocab.discard(i)
        discarded.append(i)

    # Split words if they are composed of multiple words
    def split(word):
        """
        Split a word into subwords based on vocabulary presence.
        
        This function recursively attempts to split a word into valid vocabulary tokens.
        It checks if prefixes of the word exist in the vocabulary and recursively splits
        the remaining part of the word. If multiple valid splits are found, it returns
        the split with a '<CAT>' (concatenation) token between valid subwords.
        
        Args:
            word (str): The word to be split.
        
        Returns:
            list: A list of subwords, potentially including '<CAT>' tokens to indicate
                  where the word was split.
        """
        for i in range(len(word)):
            if word[:i] in vocab:
                _ = split(word[i:])
                if len(_) > 1:
                    return [word[:i], '<CAT>'] + _
                else:
                    if word[i:] in vocab:
                        return [word[:i], '<CAT>', word[i:]]
                    continue
        return [word]
    text = [[split(word) for word in line] for line in text]
    new_text = []
    for line in text:
        new_line = []
        for word in line:
            if type(word) == list:
                new_line.extend(word)
            else:
                new_line.append(word)
        new_text.append(new_line)
        text = new_text

    # Change exceptions
    for line in range(len(text)):
        for exception in exceptions.exceptions.keys():
            text[line] = ' '.join(text[line]).replace(exception, exceptions.exceptions[exception]).split()
    
    # Build a new vocab
    vocab = set(' '.join([' '.join(line) for line in text]).split())
    vocab.add('<CAT>')
    vocab.add('<NUM>')
    vocab.add('<UNK>')
    vocab.add('<PAD>')
    vocab.add('<SOS>')
    vocab.add('<EOS>')

    # Load the vocab if it exists and add new words, because wer wanna keep words sort, because words index is important in saved emmbeddings
    if os.path.exists('data/vocab.txt'):
        with open('data/vocab.txt', 'r', encoding='utf-8') as f:
            vocab_ = [word[:-1] for word in f.readlines()]
        vocab = vocab_ + [word for word in vocab if word not in vocab_]
    
    return text, vocab

class TextDataset(Dataset):
    """
    A PyTorch Dataset for processing and indexing text data with vocabulary mapping.
    
    This dataset handles text preprocessing, including:
    - Sentence tokenization
    - Vocabulary indexing
    - Padding sequences to a fixed max length
    - Optional tensor saving/loading
    
    Attributes:
        text (list): Tokenized sentences
        vocab (list): Vocabulary list
        max_length (int): Maximum sequence length
        vocab_size (int): Size of the vocabulary
        tensor (torch.Tensor): Indexed and padded text tensor
    
    Methods:
        __getitem__: Returns input and target sequences for model training
        __len__: Returns total number of sequences in the dataset
    """

    def __init__(self, text, max_length, vocab = vocab, save_path = None):
        """
        Initialize a TextDataset with tokenized text and vocabulary mapping.
        
        Args:
            text (list): Input text data to be processed
            max_length (int): Maximum sequence length for padding and truncation
            vocab (list, optional): Vocabulary list for word indexing. Defaults to global vocab.
            save_path (str, optional): Path to save or load preprocessed tensor data
        
        Processes text by:
            - Tokenizing sentences
            - Indexing words using vocabulary
            - Padding or truncating sequences
            - Optionally saving/loading preprocessed tensor
        """
        self.text = nltk.sent_tokenize(' '.join(text))
        self.vocab = list(vocab)
        self.max_length = max_length
        self.vocab_size = len(vocab)
        if save_path is not None:
            if os.path.exists(save_path):
                self.tensor = torch.load(save_path)
                print('Data loaded from', save_path)
                print('Tensor shape:', self.tensor.shape)
                return
        self.inexed_text = []
        print('Indexing words:')
        for line in tqdm.tqdm(self.text):
            self.inexed_text.append([self.vocab.index('<SOS>')])
            for word in line.split()[:max_length]:
                if word in self.vocab:
                    self.inexed_text[-1].append(self.vocab.index(word))
                else:
                    self.inexed_text[-1].append(self.vocab.index('<UNK>'))
            self.inexed_text[-1].append(self.vocab.index('<EOS>'))
        self.shaped_text = []
        print('Convert data to tensor:')
        for line in tqdm.tqdm(self.inexed_text):
            if len(line) > self.max_length:
                self.shaped_text.append(line[:self.max_length])
            else:
                self.shaped_text.append(line + [self.vocab.index('<PAD>') for _ in range(self.max_length - len(line))])
        self.tensor = torch.tensor(np.array(self.shaped_text))
        print('Tensor shape:', self.tensor.shape)
        if save_path is not None:
            torch.save(self.tensor, save_path)

    def __getitem__(self, index):
        return self.tensor[index, :-1], self.tensor[index, 1:]
        
    def __len__(self):
        return len(self.text)

class FanmtlNovel(TextDataset):
    """
    A dataset class for loading and processing novel text data from fanmtl.com.
    
    Attributes:
        max_length (int): Maximum sequence length for text processing
        vocab (list): Vocabulary list for text indexing
        save_path (str, optional): Path to save processed tensor data
    
    Methods:
        web_scrape(num_pages): Scrapes chapters from a specific novel website
            Args:
                num_pages (int): Number of chapters to scrape
            Returns:
                list: Scraped chapter texts in lowercase
    """

    def __init__(self, max_length, vocab=vocab, save_path=None):
        """
        Initialize the Novel dataset by loading or generating text data.
        
        This method checks for existing processed or raw text files, and if not found,
        scrapes novel chapters. It processes the text and saves it to files.
        
        Args:
            max_length (int): Maximum sequence length for text processing
            vocab (list, optional): Vocabulary list for text indexing. Defaults to global vocab.
            save_path (str, optional): Path to save processed tensor data. Defaults to None.
        """
        if os.path.exists('data/processed_novel.txt'):
            with open('data/processed_novel.txt', 'r', encoding='utf-8') as f:
                text = f.readlines()
        elif os.path.exists('data/novel.txt'):
            with open('data/novel.txt', 'r', encoding='utf-8') as f:
                text = f.readlines()
            text, _ = process(text)
            text = [' '.join(t) + '\n' for t in text]
            with open('data/processed_novel.txt', 'w', encoding='utf-8') as f:
                f.writelines(text)
        else:
            text = self.web_scrape(1000)
            with open('data/novel.txt', 'w', encoding='utf-8') as f:
                f.writelines([t + '\n' for t in text])
            text, _ = process(text)
            text = [' '.join(t) + '\n' for t in text]
            with open('data/processed_novel.txt', 'w', encoding='utf-8') as f:
                f.writelines(text)
        super().__init__(text, max_length, vocab, save_path)
    
    def web_scrape(self, num_pages) -> list[str]: # TODO: make this better, it just scrapes the first 1000 chapters of "shadow slave" novel
        """
        Scrape chapters from the Shadow Slave novel on fanmtl.com.
        
        This method retrieves chapters from the specified novel website, handling rate limiting and converting text to lowercase.
        
        Args:
            num_pages (int): Number of chapters to scrape from the novel
        
        Returns:
            list: A list of scraped chapter texts in lowercase
        """
        def get_chapter(url):
            while True:
                html = requests.get(url)
                if html.status_code != 429: break # Error 429 means too many requests
                else: time.sleep(5)
            soup = BeautifulSoup(html.text, 'html.parser')
            text = soup.find("div", class_="chapter-content").text
            return text.lower()
        chaps = ['https://www.fanmtl.com/novel/shadow-slave_' + str(i) + '.html' for i in range(1, num_pages)]
        chapters_text = []
        print('Scraping chapters:')
        for i in tqdm.tqdm(chaps): 
            chapters_text.append(get_chapter(i))
        return chapters_text

class Wikipedia(TextDataset):
    """
    Wikipedia Dataset for Text Processing
    
    A TextDataset subclass that scrapes and processes Wikipedia pages for text analysis.
    
    Attributes:
        max_length (int): Maximum sequence length for text processing
        vocab (list): Vocabulary list for text indexing
        save_path (str, optional): Path to save processed tensor data
    
    Methods:
        web_scrape(num_pages, start_page): Scrapes random Wikipedia pages
            Args:
                num_pages (int): Number of pages to scrape
                start_page (str, optional): Starting URL for web scraping, defaults to random Wikipedia page
            Returns:
                list: Processed text content from scraped Wikipedia pages
    """

    def __init__(self, max_length, vocab=vocab, save_path=None):
        """
        Initialize the Wikipedia dataset, processing text from existing files or scraping new content.
    
        This method checks for existing processed or raw Wikipedia text files. If not found, it scrapes Wikipedia pages, processes the text, and saves the results.
        
        Args:
            max_length (int): Maximum sequence length for text processing
            vocab (list, optional): Vocabulary list for text indexing. Defaults to global vocab.
            save_path (str, optional): Path to save processed tensor data. Defaults to None.
        """
        if os.path.exists('data/processed_wikipedia.txt'):
            with open('data/processed_wikipedia.txt', 'r', encoding='utf-8') as f:
                text = f.readlines()
        elif os.path.exists('data/wikipedia.txt'):
            with open('data/wikipedia.txt', 'r', encoding='utf-8') as f:
                text = f.readlines()
            text, _ = process(text)
            text = [' '.join(t) + '\n' for t in text]
            with open('data/processed_wikipedia.txt', 'w', encoding='utf-8') as f:
                f.writelines(text)
        else:
            text = self.web_scrape(1000)
            with open('data/wikipedia.txt', 'w', encoding='utf-8') as f:
                f.writelines([t + '\n' for t in text])
            text, _ = process(text)
            text = [' '.join(t) + '\n' for t in text]
            with open('data/processed_wikipedia.txt', 'w', encoding='utf-8') as f:
                f.writelines(text)
        super().__init__(text, max_length, vocab, save_path)

    def web_scrape(self, num_pages, start_page = 'https://en.wikipedia.org/wiki/Special:Random') -> list[str]:
        """
        Scrape Wikipedia pages and collect text content.
        
        Retrieves a specified number of Wikipedia pages, starting from a random page or a given URL.
        Avoids duplicate pages and filters out short or irrelevant content.
        
        Args:
            num_pages (int): Number of Wikipedia pages to scrape
            start_page (str, optional): Initial Wikipedia page URL. Defaults to a random Wikipedia page.
        
        Returns:
            list: Processed text content from scraped Wikipedia pages
        """
        url = start_page
        pages = []
        if os.path.exists('data/wikipedia_pages.txt'):
            with open('data/wikipedia_pages.txt', 'r', encoding='utf-8') as f:
                pages = f.readlines()
            pages = [page[:-1] for page in pages] # Remove newline character
        texts = []
        print('Scraping wikipedia pages:')
        for page in tqdm.tqdm(range(num_pages)):
            while True:
                html = requests.get(url)
                if html.status_code != 429: break # Error 429 means too many requests
                else: time.sleep(5)
            pages.append(url)
            soup = BeautifulSoup(html.text, 'html.parser')
            content = soup.find("div", id='bodyContent')
            links = [link.get('href') for link in content.find_all('a') if link.get('href') and not link.get('href') in pages and link.get('href').startswith('/wiki/') and ':' not in link.get('href')]
            if not links:
                url = start_page
                continue
            url = random.choice(links)
            url = 'https://en.wikipedia.org' + url
            text = content.find_all('p')
            text = [line.text for line in text]
            text = ' '.join(text)
            if len(text) < 100:
                continue
            text = filter(text)
            text = re.sub(r'\[.*?\]', '', text)
            texts.append(text)
        with open('data/wikipedia_pages.txt', 'w', encoding='utf-8') as f:
            f.writelines([page + '\n' for page in pages])
        return texts

# TODO: add more datasets
