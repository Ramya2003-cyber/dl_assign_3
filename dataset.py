from collections import Counter
import spacy
from spacy.cli import download as spacy_download
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ── Special token indices ─────────────────────────────────────────────
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']


# ══════════════════════════════════════════════════════════════════════
#  CUSTOM VOCABULARY
# ══════════════════════════════════════════════════════════════════════

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

        # 2. Sort by frequency descending, then alphabetically for ties
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        for tok, freq in sorted_tokens:
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
                raise KeyError(f"Token '{tok}' not in vocab and no default index set.")
        return indices

    def __len__(self):
        return len(self.itos)


# ══════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset, tokenises all sentences once at
        construction time, and stores raw token lists for reuse by
        build_vocab() and process_data().
        """
        self.split = split
        self.special_tokens = SPECIAL_TOKENS
        self.unk_idx = UNK_IDX
        self.pad_idx = PAD_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        self.data = []

        # ── Load spaCy models, auto-download if missing ───────────────
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy models not found — downloading now...")
            spacy_download("de_core_news_sm")
            spacy_download("en_core_web_sm")
            self.spacy_de = spacy.load("de_core_news_sm")
            self.spacy_en = spacy.load("en_core_web_sm")

        # ── Load raw sentence pairs ───────────────────────────────────
        self.dataset = load_dataset("bentrevett/multi30k", split=self.split)

        # ── Tokenise once; reused by both build_vocab & process_data ──
        self.de_tokens = [
            [tok.text.lower() for tok in self.spacy_de.tokenizer(ex['de'])]
            for ex in self.dataset
        ]
        self.en_tokens = [
            [tok.text.lower() for tok in self.spacy_en.tokenizer(ex['en'])]
            for ex in self.dataset
        ]

    def build_vocab(self, min_freq=2):
        """
        Builds German and English vocabularies from the already-tokenised
        sentences. Always counts from the training split's token lists —
        no second dataset download needed.
        """
        de_counter = Counter(tok for sent in self.de_tokens for tok in sent)
        en_counter = Counter(tok for sent in self.en_tokens for tok in sent)

        self.de_vocab = CustomVocab(de_counter, min_freq=min_freq, specials=self.special_tokens)
        self.en_vocab = CustomVocab(en_counter, min_freq=min_freq, specials=self.special_tokens)
        self.de_vocab.set_default_index(self.unk_idx)
        self.en_vocab.set_default_index(self.unk_idx)

    def process_data(self):
        """
        Converts the pre-tokenised sentences into LongTensor index sequences.
        Must be called after build_vocab() (or after assigning de_vocab/en_vocab
        from a training split).
        """
        # Guard: catch missing vocab early with a clear message
        if not hasattr(self, 'de_vocab') or not hasattr(self, 'en_vocab'):
            raise RuntimeError(
                "Vocabularies not found. Call build_vocab() first, or assign "
                "de_vocab and en_vocab from the training dataset."
            )

        self.data = []
        for de_toks, en_toks in zip(self.de_tokens, self.en_tokens):
            de_indices = [self.sos_idx] + self.de_vocab.lookup_indices(de_toks) + [self.eos_idx]
            en_indices = [self.sos_idx] + self.en_vocab.lookup_indices(en_toks) + [self.eos_idx]
            self.data.append((
                torch.tensor(de_indices, dtype=torch.long),
                torch.tensor(en_indices, dtype=torch.long),
            ))

    def collate_fn(self, batch):
        """Pad a batch of variable-length (src, tgt) pairs to the longest sequence."""
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=self.pad_idx)
        return src_padded, tgt_padded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]