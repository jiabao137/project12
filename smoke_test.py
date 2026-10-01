"""
Day 1 smoke test — run this directly in PyCharm (right-click -> Run).

Checks that every image in TEST_DIR can be:
  1. Found and its filename parsed correctly (bbox + plate number)
  2. Loaded, cropped, and resized without error
  3. Turned into a valid (3, 32, 100) tensor with a 7-character label

Prints one line per image so you can eyeball whether the parsed plate
numbers look like real plates. No training, no model, just data loading.
"""

import os
from dataset import CCPDDataset, parse_ccpd_filename

# ---------------------------------------------------------------------------
# EDIT THIS if your folder is somewhere else. Keep the r"..." (raw string) —
# it stops Windows backslashes from being misread as escape characters.
# ---------------------------------------------------------------------------
TEST_DIR = r"D:\smoke test"


def main():
    print(f"Looking in: {TEST_DIR}")
    if not os.path.isdir(TEST_DIR):
        print(f"!! Folder does not exist: {TEST_DIR}")
        print("!! Double-check the path above and edit TEST_DIR at the top of this file.")
        return

    jpgs = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
    print(f"Found {len(jpgs)} .jpg files.\n")

    # --- Step 1: filename parsing only (fast, catches format issues early) ---
    print("=== Step 1: filename parsing ===")
    parse_failures = []
    for fname in jpgs:
        result = parse_ccpd_filename(os.path.join(TEST_DIR, fname))
        if result is None:
            parse_failures.append(fname)
        else:
            plate, bbox, vertices = result
            has_verts = "yes" if vertices is not None else "no "
            print(f"  OK   plate={plate:9s}  bbox={bbox}  vertices={has_verts}  file={fname}")

    if parse_failures:
        print(f"\n!! {len(parse_failures)} filename(s) failed to parse:")
        for f in parse_failures:
            print(f"     {f}")
    else:
        print(f"\nAll {len(jpgs)} filenames parsed successfully.")

    # --- Step 2: full dataset loading (crop, resize, tensor conversion) ---
    print("\n=== Step 2: full image loading (crop + resize + tensor) ===")
    try:
        ds = CCPDDataset(TEST_DIR)
    except Exception as e:
        print(f"!! CCPDDataset failed to initialise: {e}")
        return

    print(f"Dataset loaded {len(ds)} valid samples.\n")

    load_failures = []
    for i in range(len(ds)):
        try:
            img, label, plate = ds[i]
            shape_ok = tuple(img.shape) == (3, 32, 100)
            label_ok = len(label) == 7
            status = "OK" if (shape_ok and label_ok) else "!! UNEXPECTED SHAPE"
            print(f"  {status:20s} idx={i:3d}  plate={plate:9s}  "
                  f"img_shape={tuple(img.shape)}  label_len={len(label)}")
            if not (shape_ok and label_ok):
                load_failures.append(i)
        except Exception as e:
            print(f"  !! FAILED idx={i:3d}: {e}")
            load_failures.append(i)

    print("\n=== Summary ===")
    print(f"Filenames parsed OK : {len(jpgs) - len(parse_failures)} / {len(jpgs)}")
    print(f"Images loaded OK    : {len(ds) - len(load_failures)} / {len(ds)}")
    if not parse_failures and not load_failures:
        print("\nAll checks passed — real CCPD-Base data loads correctly.")
    else:
        print("\nSome checks failed — copy the error lines above and send them over.")


if __name__ == "__main__":
    main()
