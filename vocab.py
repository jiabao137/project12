"""
CCPD plate character vocabulary.

CCPD encodes each plate as 7 tokens: [province, alphabet, ad, ad, ad, ad, ad]
where the three fields below are index lookup tables into the filename encoding
(see dataset.py: parse_ccpd_filename). This module builds the CTC vocabulary as
the union of all characters that can appear, matching the vocab ~= 70 (+1 blank)
setup used elsewhere in this project.

Index 0 is reserved for the CTC blank symbol.
"""

PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "警", "学", "O",
]

ALPHABETS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "O",
]

ADS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O",
]

BLANK = "<blank>"

# Build the CTC vocabulary as the sorted union of every symbol that can appear,
# with the CTC blank fixed at index 0.
_unique_chars = sorted(set(PROVINCES) | set(ALPHABETS) | set(ADS))
CHARS = [BLANK] + _unique_chars

CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

NUM_CLASSES = len(CHARS)   # ~= 71 (blank + ~70 symbols); print at import time to verify
BLANK_IDX = CHAR2IDX[BLANK]

PLATE_LENGTH = 7  # every CCPD plate is exactly 7 characters


def encode_plate(plate_str):
    """'皖A12345' -> [idx, idx, ...] (length 7)"""
    return [CHAR2IDX[c] for c in plate_str]


def decode_indices(indices):
    """[idx, idx, ...] -> '皖A12345', skipping any blank indices."""
    return "".join(IDX2CHAR[i] for i in indices if i != BLANK_IDX)


if __name__ == "__main__":
    print(f"Vocabulary size (incl. blank): {NUM_CLASSES}")
    print(f"Blank index: {BLANK_IDX}")
