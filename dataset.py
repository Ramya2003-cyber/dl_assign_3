from collections import Counter
import spacy
from datasets import load_dataset
import torch

class CustomVocab:
    def __init__(self, counter, min_freq=1, specials=None):
        self.stoi = {}  # String to Index
        self.itos = []  # Index to String
        self.default_index = None

        # 1. Add special tokens first (so they get indices 0, 1, 2, 3)
        if specials is not None:
            for tok in specials:
                self.itos.append(tok)
                self.stoi[tok] = len(self.itos) - 1

        # 2. Add words that meet the minimum frequency
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self.stoi:
                self.itos.append(tok)
                self.stoi[tok] = len(self.itos) - 1

    def set_default_index(self, index):
        self.default_index = index

    def lookup_indices(self, tokens):
        indices = []
        for tok in tokens:
            if tok in self.stoi:
                indices.append(self.stoi[tok])
            elif self.default_index is not None:
                indices.append(self.default_index)
            else:
                raise KeyError(f"Token {tok} not found.")
        return indices

    def __len__(self):
        return len(self.itos)

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        self.dataset = load_dataset("bentrevett/multi30k",split=self.split)  # Placeholder for the loaded dataset
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")
        self.special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
        self.unk_idx = 0 #unkown
        self.pad_idx = 1 #padding
        self.sos_idx=2 #start of sentence
        self.eos_idx=3 #end of sentence
    

    def build_vocab(self,min_freq=2):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        train_data=load_dataset("bentrevett/multi30k",split='train')
        de_counter=Counter() #dictionaries to count the frequency of each token in the German and English
        en_counter=Counter() 
        for example in train_data:
            de_counter.update([token.text.lower() for token in self.spacy_de.tokenizer(example['de'])]) #updating every token in the couner
            en_counter.update([token.text.lower() for token in self.spacy_en.tokenizer(example['en'])])
        
        self.de_vocab = CustomVocab(de_counter, min_freq=min_freq, specials=self.special_tokens)
        self.en_vocab = CustomVocab(en_counter, min_freq=min_freq, specials=self.special_tokens)
        self.de_vocab.set_default_index(self.unk_idx) #setting the default index for unknown tokens to the unk_idx
        self.en_vocab.set_default_index(self.unk_idx)


    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        self.data=[]
        for example in self.dataset:
            de_tokens=[token.text.lower() for token in self.spacy_de.tokenizer(example['de'])]
            en_tokens=[token.text.lower() for token in self.spacy_en.tokenizer(example['en'])]
            de_indices=self.de_vocab.lookup_indices(de_tokens) #converting the tokens to indices using the lookup_indices method of the vocab
            en_indices=self.en_vocab.lookup_indices(en_tokens)
            de_indices=[self.sos_idx] + de_indices + [self.eos_idx]
            en_indices=[self.sos_idx] + en_indices + [self.eos_idx]
            de_tensor=torch.tensor(de_indices,dtype=torch.long) #converting the indices to tensors
            en_tensor=torch.tensor(en_indices,dtype=torch.long)
            self.data.append((de_tensor,en_tensor)) #appending the tuple of tensors to the data list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]