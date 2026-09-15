"""
Fonctions et classes partagées entre les notebooks du projet POS Tagging Bambara.
Chaque notebook fait :
    import sys
    from pathlib import Path
    sys.path.append(str(Path.cwd().parent))
    from bambara_pos_utils import *
"""

import re
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------------------------
# 01 — Tokenisation et vocabulaire
# ---------------------------------------------------------------------------

def bamanankan_tokenizer(text):
    """Tokenise une phrase en bambara (mots + ponctuation séparée)."""
    text = text.strip()
    pattern = r"[a-zA-ZÀ-ÿɛƐɔƆɲƝŋŊ]+|[.,!?'\";:()\[\]]"
    return re.findall(pattern, text)


def build_vocab(data):
    """data : liste de tuples (mots, tags). Retourne word_to_ix, tag_to_ix."""
    word_to_ix = {"<PAD>": 0, "<UNK>": 1}
    tag_to_ix = {"<PAD>": 0}
    for words, tags in data:
        for word in words:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    return word_to_ix, tag_to_ix


def prepare_sequence(seq, to_ix):
    """Convertit une liste de tokens en tenseur d'index, repli sur <UNK>."""
    idxs = [to_ix.get(w, to_ix.get("<UNK>", 1)) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Tagset fixe du projet (utilisé partout à la place d'un tag_to_ix reconstruit)
VALID_TAGS = ["<PAD>", "ADJ", "ADV", "AUX", "CONJ", "DET", "NOM",
              "PART", "POSTP", "PRON", "PUNCT", "VERBE"]
label2id = {tag: i for i, tag in enumerate(VALID_TAGS)}
id2label = {i: tag for tag, i in label2id.items()}


# ---------------------------------------------------------------------------
# 02 — Modèle
# ---------------------------------------------------------------------------

class LSTMTagger(nn.Module):
    """Version pédagogique : une seule phrase à la fois."""
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(tag_space, dim=1)


class LSTMTaggerWithBatch(nn.Module):
    """Version batchée, utilisée pour l'entraînement réel (BiLSTM)."""
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, padding_idx=0):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentences):
        embeds = self.word_embeddings(sentences)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return torch.log_softmax(tag_space, dim=-1)


# ---------------------------------------------------------------------------
# 03 — Dataset, batching, entraînement
# ---------------------------------------------------------------------------

class BambaraPOSDataset(Dataset):
    def __init__(self, data, word_to_ix, tag_to_ix):
        self.data = data
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = self.data[idx]
        word_ids = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w in words]
        tag_ids = [self.tag_to_ix.get(t, 0) for t in tags]
        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long)


def collate_fn_padd(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        tag_scores = model(inputs)
        b, s, t = tag_scores.shape
        loss = loss_fn(tag_scores.view(b * s, t), targets.view(b * s))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, tagset_size, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            tag_scores = model(inputs)
            b, s, t = tag_scores.shape
            tag_scores_flat = tag_scores.view(b * s, t)
            targets_flat = targets.view(b * s)
            loss = loss_fn(tag_scores_flat, targets_flat)
            total_loss += loss.item()
            preds = torch.argmax(tag_scores_flat, dim=1)
            mask = targets_flat != 0
            correct += (preds[mask] == targets_flat[mask]).sum().item()
            total += mask.sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# 04 — Chargement du corpus et lexique
# ---------------------------------------------------------------------------

def load_bambara_corpus(file_path):
    """Lit un fichier CoNLL/TSV (mot <TAB> tag, ligne vide = fin de phrase)."""
    sentences = []
    current_words, current_tags = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                if current_words:
                    assert len(current_words) == len(current_tags), \
                        f"Désalignement mots/tags à la ligne {line_num}"
                    sentences.append((current_words, current_tags))
                    current_words, current_tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                current_words.append(parts[0])
                current_tags.append(parts[1])
            else:
                print(f"Ligne {line_num} ignorée (format inattendu) : {line!r}")
        if current_words:
            sentences.append((current_words, current_tags))
    print(f"Corpus chargé : {len(sentences)} phrases.")
    return sentences


BAMBARA_EXTENDED_LEXICON = {
    'a': 'PRON', 'i': 'PRON', 'u': 'PRON', 'n': 'PRON', 'ne': 'PRON',
    'an': 'PRON', 'aw': 'PRON', 'e': 'PRON', 'olu': 'PRON', 'ale': 'PRON',
    'min': 'PRON', 'minnu': 'PRON', 'yɛrɛ': 'PRON',
    'ka': 'AUX', 'ye': 'AUX', 'bɛ': 'AUX', 'tɛ': 'AUX', 'ma': 'AUX',
    'tun': 'AUX', 'be': 'AUX', 'te': 'AUX', 'b': 'AUX', 'y': 'AUX', 'k': 'AUX',
    'la': 'POSTP', 'na': 'POSTP', 'kan': 'POSTP', 'fɛ': 'POSTP',
    'kɔnɔ': 'POSTP', 'bolo': 'POSTP', 'kɔ': 'POSTP',
    'o': 'DET', 'nin': 'DET', 'dɔ': 'DET', 'si': 'DET', 'bɛɛ': 'DET', 'in': 'DET',
    'ani': 'CONJ', 'ni': 'CONJ', 'nka': 'CONJ', 'ko': 'CONJ',
    'de': 'PART', 'don': 'PART', 'dɔrɔn': 'PART',
    'fana': 'ADV', 'yen': 'ADV',
    'se': 'VERBE', 'sɔrɔ': 'VERBE', 'bɔ': 'VERBE', 'taa': 'VERBE', 'to': 'VERBE',
    'dɔn': 'VERBE', 'fɔ': 'VERBE', 'di': 'VERBE', 'da': 'VERBE', 'kɛ': 'VERBE',
    'nana': 'VERBE', 'fo': 'VERBE',
    'ala': 'NOM', 'mɔgɔ': 'NOM', 'cɛ': 'NOM', 'fɛn': 'NOM', 'dugu': 'NOM',
    'cogo': 'NOM', 'ɲɔgɔn': 'NOM', 'tuma': 'NOM', 'yezu': 'NOM',
    'kelen': 'ADJ',
}

PUNCT_SET = {',', '.', "'", ':', ')', '(', '!', ';', '?', '"', '-', '['}


def tag_with_lexicon(tok, prev_tag):
    if tok in PUNCT_SET:
        return 'PUNCT'
    low = tok.lower()
    if low in BAMBARA_EXTENDED_LEXICON:
        return BAMBARA_EXTENDED_LEXICON[low]
    if prev_tag == 'AUX':
        return 'VERBE'
    return 'NOM'


def enrich_tags_with_lexicon(corpus):
    enriched = []
    for words, _ in corpus:
        new_tags, prev_tag = [], None
        for w in words:
            tag = tag_with_lexicon(w, prev_tag)
            new_tags.append(tag)
            prev_tag = tag
        enriched.append((words, new_tags))
    return enriched


def predict_hybrid(tokens, model, word_to_ix, id2label, lexicon, device):
    results, unresolved_idx = [], []
    for i, tok in enumerate(tokens):
        if tok in PUNCT_SET:
            results.append((tok, "PUNCT"))
        elif tok.lower() in lexicon:
            results.append((tok, lexicon[tok.lower()]))
        else:
            results.append((tok, None))
            unresolved_idx.append(i)

    if unresolved_idx:
        model.eval()
        ids = [word_to_ix.get(t, word_to_ix["<UNK>"]) for t in tokens]
        input_tensor = torch.tensor([ids], dtype=torch.long).to(device)
        with torch.no_grad():
            preds = torch.argmax(model(input_tensor), dim=-1).squeeze(0).tolist()
        for idx in unresolved_idx:
            results[idx] = (tokens[idx], id2label[preds[idx]])
    return results

import random
import json


def split_corpus(corpus_data, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split reproductible train/val/test. Le seed=42 est partagé entre tous les
    notebooks du projet : si tu changes cette valeur ici, refais tourner 05 et 06
    pour rester cohérent, sinon les deux notebooks n'évaluent plus sur le même test set.
    """
    data = corpus_data.copy()
    random.Random(seed).shuffle(data)
    n = len(data)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


def save_checkpoint(model, word_to_ix, hyperparams, path):
    """Sauvegarde le modèle + son vocabulaire + ses hyperparamètres dans un seul fichier."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "word_to_ix": word_to_ix,
        "hyperparams": hyperparams,
    }, path)


def load_checkpoint(path, device):
    """Recharge un modèle entraîné, prêt pour l'inférence."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hp = checkpoint["hyperparams"]
    model = LSTMTaggerWithBatch(
        hp["embedding_dim"], hp["hidden_dim"], len(checkpoint["word_to_ix"]), len(label2id)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["word_to_ix"], hp


def per_tag_f1(report_dict):
    """
    À partir d'un classification_report(..., output_dict=True), retourne
    un dict {tag: f1-score} trié du pire au meilleur (hors <PAD>).
    """
    scores = {
        tag: vals["f1-score"] for tag, vals in report_dict.items()
        if tag in VALID_TAGS and tag != "<PAD>" and isinstance(vals, dict)
    }
    return dict(sorted(scores.items(), key=lambda x: x[1]))


# --- Ajouts pour le module 07 : SentencePiece ---

from collections import Counter as _Counter


def export_raw_text(corpus_data, output_path):
    """Extrait uniquement les mots (sans les tags) pour entraîner SentencePiece."""
    with open(output_path, "w", encoding="utf-8") as f:
        for words, _ in corpus_data:
            f.write(" ".join(words) + "\n")


def encode_sentence_with_tags(words, tags, sp, scheme="broadcast"):
    """
    Découpe chaque mot en sous-tokens SentencePiece et propage son tag.
    scheme="broadcast" : chaque sous-token reçoit le tag du mot entier.
    scheme="bio"        : premier sous-token -> B-TAG, suivants -> I-TAG.
    """
    subword_tokens, subword_tags = [], []
    for word, tag in zip(words, tags):
        pieces = sp.encode(word, out_type=str)
        if not pieces:
            continue
        for i, piece in enumerate(pieces):
            subword_tokens.append(piece)
            if scheme == "broadcast":
                subword_tags.append(tag)
            elif scheme == "bio":
                subword_tags.append(f"B-{tag}" if i == 0 else f"I-{tag}")
    return subword_tokens, subword_tags


def build_bio_tagset(valid_tags):
    """Double le tagset : B-TAG et I-TAG pour chaque tag, en gardant <PAD>."""
    bio_tags = ["<PAD>"]
    for tag in valid_tags:
        if tag == "<PAD>":
            continue
        bio_tags += [f"B-{tag}", f"I-{tag}"]
    label2id_bio = {t: i for i, t in enumerate(bio_tags)}
    id2label_bio = {i: t for t, i in label2id_bio.items()}
    return label2id_bio, id2label_bio


def _resolve_word_tag(tags_for_word, scheme):
    if scheme == "bio":
        return tags_for_word[0].replace("B-", "").replace("I-", "")
    return _Counter(tags_for_word).most_common(1)[0][0]


def reconstruct_word_tags(subword_tokens, predicted_tags, scheme="broadcast"):
    """
    Regroupe les sous-tokens par mot (un nouveau mot commence à chaque '▁')
    et déduit un seul tag par mot à partir des prédictions sous-token.
    """
    words_tags, current_tags = [], []
    for token, tag in zip(subword_tokens, predicted_tags):
        if token.startswith("▁") and current_tags:
            words_tags.append(_resolve_word_tag(current_tags, scheme))
            current_tags = []
        current_tags.append(tag)
    if current_tags:
        words_tags.append(_resolve_word_tag(current_tags, scheme))
    return words_tags


class BambaraSubwordDataset(Dataset):
    """Équivalent de BambaraPOSDataset, mais au niveau sous-mot (SentencePiece)."""
    def __init__(self, data, sp, tag_to_ix, scheme="broadcast"):
        self.data = data
        self.sp = sp
        self.tag_to_ix = tag_to_ix
        self.scheme = scheme

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = self.data[idx]
        subword_tokens, subword_tags = encode_sentence_with_tags(
            words, tags, self.sp, self.scheme
        )
        token_ids = [self.sp.piece_to_id(t) for t in subword_tokens]
        tag_ids = [self.tag_to_ix.get(t, 0) for t in subword_tags]
        return (torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(tag_ids, dtype=torch.long))


def run_training(train_data, val_data, dataset_cls, dataset_args, vocab_size,
                  tagset_size, hyperparams, device,
                  patience=3, max_epochs=30, checkpoint_path=None, verbose=True):
    """
    Boucle d'entraînement/validation générique avec early stopping et sauvegarde
    du meilleur modèle. dataset_cls/dataset_args permettent de réutiliser cette
    même fonction pour BambaraPOSDataset (mot-entier) ET BambaraSubwordDataset.
    """
    train_loader = DataLoader(
        dataset_cls(train_data, *dataset_args),
        batch_size=hyperparams["batch_size"], shuffle=True, collate_fn=collate_fn_padd
    )
    val_loader = DataLoader(
        dataset_cls(val_data, *dataset_args),
        batch_size=hyperparams["batch_size"], shuffle=False, collate_fn=collate_fn_padd
    )

    model = LSTMTaggerWithBatch(
        hyperparams["embedding_dim"], hyperparams["hidden_dim"], vocab_size, tagset_size
    ).to(device)

    train_labels_flat = [t for _, tags in train_data for t in tags]
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=0.01)
    loss_fn = nn.NLLLoss(ignore_index=0)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, tagset_size, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"Époque {epoch+1:2d} | Train Loss : {train_loss:.4f} | "
                  f"Val Loss : {val_loss:.4f} | Val Acc : {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                torch.save({"model_state_dict": model.state_dict(), "hyperparams": hyperparams}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Arrêt anticipé à l'époque {epoch+1}.")
                break

    return model, history, best_val_loss

def evaluate_subword_model_word_level(model, sentences, sp, id2label, scheme, device):
    """
    Évalue un modèle entraîné sur des sous-tokens SentencePiece, au niveau mot.
    Reconstruit un tag par mot à partir des prédictions sous-token avant de comparer
    aux vrais tags, pour rester comparable aux évaluations mot-entier (05/06).
    """
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for words, tags in sentences:
            subword_tokens, _ = encode_sentence_with_tags(words, tags, sp, scheme)
            if not subword_tokens:
                continue
            ids = [sp.piece_to_id(t) for t in subword_tokens]
            input_tensor = torch.tensor([ids], dtype=torch.long).to(device)
            preds = torch.argmax(model(input_tensor), dim=-1).squeeze(0).cpu().tolist()
            predicted_tags = [id2label[p] for p in preds]

            reconstructed = reconstruct_word_tags(subword_tokens, predicted_tags, scheme)
            n = min(len(reconstructed), len(tags))
            all_true.extend(tags[:n])
            all_pred.extend(reconstructed[:n])

    accuracy = accuracy_score(all_true, all_pred)
    report = classification_report(all_true, all_pred, zero_division=0)
    return accuracy, report