#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from collections import Counter, defaultdict
import torch

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self, order=4, max_vocab=5000):
        self.order = int(order)
        self.max_vocab = int(max_vocab)

        # vocab
        self.itos = []
        self.stoi = {}
        self.unk_id = None

        # ngram counts:
        # context(tuple[int]) -> Counter(next_id -> count)
        self.ngram = defaultdict(Counter)
        self.unigram = Counter()

        # fallback chars if everything is empty
        self.fallback_chars = [" ", "e", "a"]

    @classmethod
    def load_training_data(cls):
        """
        Loads a small amount of OSCAR text for basic functionality.

        Environment variables (optional):
          - OSCAR_LANGS: comma-separated language codes (default: "en")
          - OSCAR_SAMPLES: number of documents to read total (default: 2000)
          - OSCAR_DATASET: dataset name (default: "oscar")
          - OSCAR_CONFIG_PREFIX: config prefix (default: "unshuffled_deduplicated_")
        """
        if load_dataset is None:
            # If datasets isn't available, return empty and training will no-op.
            return []

        langs = os.environ.get("OSCAR_LANGS", "en").split(",")
        langs = [l.strip() for l in langs if l.strip()]
        n_docs = int(os.environ.get("OSCAR_SAMPLES", "2000"))

        dataset_name = os.environ.get("OSCAR_DATASET", "oscar")
        config_prefix = os.environ.get("OSCAR_CONFIG_PREFIX", "unshuffled_deduplicated_")

        # Stream and collect texts (small sample for checkpoint 1)
        texts = []
        per_lang = max(1, n_docs // max(1, len(langs)))

        for lang in langs:
            config = f"{config_prefix}{lang}"
            try:
                ds = load_dataset(dataset_name, config, split="train", streaming=True)
            except Exception:
                # If a config fails, skip it (keeps run-to-spec).
                continue

            try:
                it = iter(ds)
                for _ in range(per_lang):
                    ex = next(it)
                    t = ex.get("text", "")
                    if t:
                        texts.append(t)
            except Exception:
                # Keep going even if streaming has hiccups
                continue

        return texts

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def _build_vocab(self, texts):
        # Count characters (Unicode-safe)
        c = Counter()
        for t in texts:
            c.update(t)

        # Keep top max_vocab-1 chars, reserve 1 for <unk>
        most_common = [ch for ch, _ in c.most_common(max(0, self.max_vocab - 1))]
        self.itos = ["<unk>"] + most_common
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.unk_id = 0

        # refresh fallbacks if present in vocab
        for ch in [" ", "e", "a"]:
            if ch in self.stoi:
                continue
        self.fallback_chars = [" ", "e", "a"]

    def _encode(self, ch):
        return self.stoi.get(ch, self.unk_id)

    def _decode(self, idx):
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return "<unk>"

    def run_train(self, data, work_dir):
        """
        Train a simple char-level N-gram model on provided text list.
        """
        if not data:
            # No training data available; keep model usable with defaults.
            self.itos = ["<unk>"]
            self.stoi = {"<unk>": 0}
            self.unk_id = 0
            return

        self._build_vocab(data)

        n = self.order
        for t in data:
            # Encode once
            ids = [self._encode(ch) for ch in t]
            if len(ids) < 2:
                continue

            # unigram stats
            self.unigram.update(ids)

            # n-gram stats with contexts up to n-1
            # For each position i, predict ids[i] given previous up to n-1 chars
            for i in range(len(ids)):
                next_id = ids[i]
                # build context of length 0..n-1 (we'll store all for backoff)
                start = max(0, i - (n - 1))
                ctx = tuple(ids[start:i])  # can be empty
                self.ngram[ctx][next_id] += 1

    def _top3_from_counter(self, counter):
        if not counter:
            # hard fallback
            return self.fallback_chars[:3]

        # Use torch to sort counts (lightweight "use pytorch" requirement)
        items = list(counter.items())
        ids = torch.tensor([i for i, _ in items], dtype=torch.long)
        counts = torch.tensor([c for _, c in items], dtype=torch.long)

        # topk by counts
        k = min(3, counts.numel())
        top_counts, top_pos = torch.topk(counts, k=k, largest=True, sorted=True)
        top_ids = ids[top_pos].tolist()

        chars = []
        for idx in top_ids:
            ch = self._decode(int(idx))
            # avoid returning "<unk>" if possible
            if ch == "<unk>":
                continue
            chars.append(ch)

        # If we filtered out too much, pad from unigram / fallbacks
        if len(chars) < 3:
            for idx, _ in self.unigram.most_common():
                ch = self._decode(int(idx))
                if ch != "<unk>" and ch not in chars:
                    chars.append(ch)
                if len(chars) == 3:
                    break

        while len(chars) < 3:
            for ch in self.fallback_chars:
                if ch not in chars:
                    chars.append(ch)
                if len(chars) == 3:
                    break

        return chars[:3]

    def run_pred(self, data):
        # Predict 3 next-character candidates per input line.
        preds = []
        n = self.order

        for inp in data:
            try:
                # Use longest context up to n-1, with backoff
                ids = [self._encode(ch) for ch in inp]
                best_counter = None

                # Try contexts of length n-1, n-2, ..., 0
                for L in range(min(n - 1, len(ids)), -1, -1):
                    ctx = tuple(ids[-L:]) if L > 0 else tuple()
                    counter = self.ngram.get(ctx)
                    if counter and sum(counter.values()) > 0:
                        best_counter = counter
                        break

                if best_counter is None:
                    best_counter = self.unigram

                top3 = self._top3_from_counter(best_counter)
                preds.append("".join(top3))
            except Exception:
                # Robustness: never crash on a weird sample
                preds.append("".join(self.fallback_chars[:3]))

        return preds

    def save(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        path = os.path.join(work_dir, "model.pt")

        # torch.save uses pickle; fine for checkpointing small dicts
        payload = {
            "order": self.order,
            "max_vocab": self.max_vocab,
            "itos": self.itos,
            "stoi": self.stoi,
            "unk_id": self.unk_id,
            "unigram": dict(self.unigram),
            # Convert Counters to dicts for serialization
            "ngram": {ctx: dict(cnt) for ctx, cnt in self.ngram.items()},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, work_dir):
        path = os.path.join(work_dir, "model.pt")
        if not os.path.exists(path):
            # No trained model found; return a default model
            return MyModel()

        payload = torch.load(path, map_location="cpu")
        m = MyModel(order=payload.get("order", 4), max_vocab=payload.get("max_vocab", 5000))

        m.itos = payload.get("itos", ["<unk>"])
        m.stoi = payload.get("stoi", {"<unk>": 0})
        m.unk_id = payload.get("unk_id", 0)

        m.unigram = Counter(payload.get("unigram", {}))
        m.ngram = defaultdict(Counter)
        for ctx, d in payload.get("ngram", {}).items():
            m.ngram[ctx] = Counter(d)

        return m


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
