from __future__ import annotations

from pathlib import Path


class CharTokenizer:
    def __init__(self, symbol_table, non_lang_syms=None, unk="<filler>", split_with_space=False):
        self.symbol_table = self._read_symbol_table(Path(symbol_table))
        self.id2token = {idx: token for token, idx in self.symbol_table.items()}
        self.unk = unk
        self.unk_id = self.symbol_table.get(unk)
        if self.unk_id is None:
            raise ValueError(f"unk token {unk!r} not found in {symbol_table}")
        self.split_with_space = split_with_space

    @staticmethod
    def _read_symbol_table(path: Path) -> dict[str, int]:
        table: dict[str, int] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    table[parts[0]] = int(parts[1])
        return table

    def text2tokens(self, text: str) -> list[str]:
        text = text.strip()
        if self.split_with_space:
            return [token for token in text.split() if token]
        return list(text)

    def tokens2ids(self, tokens: list[str]) -> list[int]:
        return [self.symbol_table.get(token, self.unk_id) for token in tokens]

    def text2ids(self, text: str) -> list[int]:
        return self.tokens2ids(self.text2tokens(text))

    def tokenize(self, text: str) -> tuple[list[str], list[int]]:
        tokens = self.text2tokens(text)
        return tokens, self.tokens2ids(tokens)

    def ids2tokens(self, ids: list[int]) -> list[str]:
        return [self.id2token.get(int(idx), self.unk) for idx in ids]

    def ids2text(self, ids: list[int]) -> str:
        tokens = self.ids2tokens(ids)
        if self.split_with_space:
            return " ".join(tokens)
        return "".join(tokens)
