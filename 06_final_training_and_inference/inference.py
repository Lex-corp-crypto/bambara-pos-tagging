"""
Module d'inférence autonome pour le POS Tagging du bambara.

Usage :
    from inference import BambaraPOSTagger
    tagger = BambaraPOSTagger(checkpoint_path="models/bambara_pos_best.pth")
    tagger.tag("Amadou bɛ kalan kɛ .")

Ce module ne dépend que de bambara_pos_utils.py (à la racine du projet) et
d'un checkpoint produit par 06_final_training_and_inference.ipynb.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from bambara_pos_utils import (
    bamanankan_tokenizer,
    load_checkpoint,
    predict_hybrid,
    BAMBARA_EXTENDED_LEXICON,
    id2label,
)

import torch


class BambaraPOSTagger:
    """Enveloppe le pipeline hybride (ponctuation -> lexique -> BiLSTM) en un objet simple."""

    def __init__(self, checkpoint_path, device=None, lexicon=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.word_to_ix, self.hyperparams = load_checkpoint(checkpoint_path, self.device)
        self.lexicon = lexicon or BAMBARA_EXTENDED_LEXICON

    def tag(self, sentence):
        """Prend une phrase brute (str) et retourne une liste de tuples (mot, tag)."""
        tokens = bamanankan_tokenizer(sentence)
        return predict_hybrid(tokens, self.model, self.word_to_ix, id2label, self.lexicon, self.device)

    def tag_batch(self, sentences):
        """Applique .tag() à une liste de phrases."""
        return [self.tag(s) for s in sentences]


