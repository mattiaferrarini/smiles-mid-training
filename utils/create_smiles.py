import os
import logging
import re
import json

# Load Periodic Table for Element Name validation
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERIODIC_TABLE_PATH = os.path.join(BASE_DIR, "json", "periodic_table.json")

ELEMENT_NAME_MAP = {}
try:
    with open(PERIODIC_TABLE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            ELEMENT_NAME_MAP[entry["Symbol"]] = entry["Element"].lower()
except Exception as e:
    logging.warning(f"Could not load periodic table from {PERIODIC_TABLE_PATH}: {e}")

# Regex for SMILES validation
# Matches organic subset, common elements, brackets, and special chars
# Excludes common English words that might match
ELEMENTS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]
ELEMENTS.sort(key=len, reverse=True)
AROMATICS = ["b", "c", "n", "o", "p", "s"]
AROMATICS_SET = set(AROMATICS)

# Pattern components
P_BRACKET = r"\[[^\]]+\]"
P_ELEMENTS = "|".join(ELEMENTS)
P_AROMATICS = "|".join(AROMATICS)
P_SPECIAL = r"[\d=\#\-\+\.\\\/@%\(\)]"

# Combined pattern for tokenization (order important)
# Bracket > Element > Aromatic > Special
TOKEN_PATTERN = f"({P_BRACKET}|{P_ELEMENTS}|{P_AROMATICS}|{P_SPECIAL})"

VALIDATION_PATTERN = re.compile(
    f"^(?:{P_BRACKET}|{P_ELEMENTS}|{P_AROMATICS}|{P_SPECIAL})+$"
)

EXCLUDE_WORDS = {
    "I",
    "In",
    "On",
    "No",
    "So",
    "Up",
    "A",
    "a",
    "Be",
    "Am",
    "As",
    "At",
    "Is",
    "Or",
    "Go",
    "Us",
    "By",
    "To",
    "It",
    "Re",
    "Pm",
    "Mt",
    "UV",
    "at",
}

# Map of Ambiguous Formulas -> Required previous word (lowercase)
# This solves the problem of acronyms like IF (Intermediate Frequency) vs IF (Iodine Fluoride)
AMBIGUOUS_FORMULAS = {
    "IF": ["fluoride"],  # Iodine Fluoride
    "HF": ["fluoride", "acid"],  # Hydrogen Fluoride / Hydrofluoric Acid
    "NO": ["oxide"],  # Nitric Oxide
    "CO": ["monoxide", "oxide"],  # Carbon Monoxide
}


def is_smiles(token, previous_word=None):
    clean_token = token.strip(".,;:!?\"'")
    if not clean_token:
        return False
    if clean_token in EXCLUDE_WORDS:
        return False

    # Rule: Must contain at least one capital letter OR one number
    if not (re.search(r"[A-Z]", clean_token) or re.search(r"\d", clean_token)):
        return False

    # SMILES cannot start with a number
    if clean_token[0].isdigit():
        return False

    # Ambiguous Formula Context Check (e.g. IF, HF)
    if clean_token in AMBIGUOUS_FORMULAS:
        if previous_word:
            clean_prev = previous_word.strip(".,;:!?\"'()[]{}").lower()
            # Check if previous word ends with one of the required suffixes (e.g. "monofluoride" ends with "fluoride")
            valid_context = False
            for req in AMBIGUOUS_FORMULAS[clean_token]:
                if clean_prev.endswith(req):
                    valid_context = True
                    break
            if not valid_context:
                return False
        else:
            return False

    # Single Element Context Check
    # If the token is exactly one element symbol (e.g. "Na", "O", "C")
    # It is valid ONLY if the previous word corresponds to the element name (e.g. "Sodium")
    if clean_token in ELEMENT_NAME_MAP:
        if previous_word:
            # Clean previous word (remove parens etc)
            clean_prev = previous_word.strip(".,;:!?\"'()[]{}").lower()
            expected_name = ELEMENT_NAME_MAP[clean_token]
            if clean_prev != expected_name:
                return False
        else:
            # If no previous word, reject single elements to be safe
            return False

    if (
        clean_token.endswith("s")
        and clean_token[:-1].isupper()
        and clean_token not in ELEMENTS
    ):
        # Only reject if it looks like a plain text acronym (no digits, no bonds, no brackets)
        if not re.search(r"[\d=\#\-\+\(\)\[\]]", clean_token):
            return False

    # Must contain at least one letter (heuristic to avoid matching pure numbers or symbols like "5" or ">>")
    if not re.search(r"[a-zA-Z]", clean_token):
        return False

    if not VALIDATION_PATTERN.match(clean_token):
        return False

    tokens = re.findall(TOKEN_PATTERN, clean_token)

    has_aromatic = False
    has_special_or_digit = False

    for t in tokens:
        if t in AROMATICS_SET:
            has_aromatic = True
        # Check if token is a special char, digit, or bracket (anything that is NOT an element or aromatic letter)
        if re.match(P_SPECIAL, t) or re.match(P_BRACKET, t):
            has_special_or_digit = True

    if has_aromatic and not has_special_or_digit:
        return False

    return True


def annotate_smiles(text):
    tokens = text.split()
    new_tokens = []
    smiles_count = 0
    for i, token in enumerate(tokens):
        # If already annotated, keep as is
        if "[START_SMILES]" in token and "[END_SMILES]" in token:
            new_tokens.append(token)
            continue

        previous_word = tokens[i - 1] if i > 0 else None

        # Check if the token (or stripped version) is a SMILES
        # "C[C@]12...," -> "[START_SMILES]C[C@]12...[END_SMILES],"
        # Modified regex to NOT strip brackets [] as they are essential for SMILES (e.g. [Na+])
        match = re.match(r"^([^\w\s\[\]]*)(.*?)([^\w\s\[\]]*)$", token)
        if match:
            prefix, core, suffix = match.groups()
            if is_smiles(core, previous_word):
                new_tokens.append(f"{prefix}[START_SMILES]{core}[END_SMILES]{suffix}")
                smiles_count += 1
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)

    return " ".join(new_tokens), smiles_count
