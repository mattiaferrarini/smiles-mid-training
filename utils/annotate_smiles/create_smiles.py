import os
import logging
import re
import json
import time

# Load Periodic Table for Element Name validation
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PERIODIC_TABLE_PATH = os.path.join(BASE_DIR, "json", "periodic_table.json")

ELEMENT_NAME_MAP = {}
try:
    with open(PERIODIC_TABLE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            ELEMENT_NAME_MAP[entry["Symbol"]] = entry["Element"].lower()
except Exception as e:
    logging.warning(f"Could not load periodic table from {PERIODIC_TABLE_PATH}: {e}")

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
# it is necessary to order them by length to avoid to confuse elements
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

FALSE_POSITIVE_PATTERN = re.compile(r"^(?![CNOPSFI]+$)[A-Z]{4,}$")

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
    # False Positives Blacklist
    "BIS",
    "BWB",
    "CISCO",
    "COOKBOOK",
    "CV",
    "FON",
    "HH",
    "HNH",
    "IP",
    "ISBN",
    "NOS",
    "NSF",
    "PPO",
    "SH",
    "SI",
    "SS",
    "UPS",
    "UNKNOWN",
    "WHY",
    "WWI",
    "US",
    "UK",
    "U.S",
    "W.H",
    "II",
    "III",
    "IV",
    "VI",
    "VII",
    "VIII",
    "IX",
    "[HAW93]",
    "B/c",
    "C-2",
    "C-4",
    "Sc1",
    "SS1",
    "SS2",
    "SS3",
    "SS4",
    "SS5",
    "SS6",
}

# It solves the problem of acronyms like IF (Intermediate Frequency) vs IF (Iodine Fluoride)
AMBIGUOUS_FORMULAS = {
    "IF": ["fluoride"],  # Iodine Fluoride
    "HF": ["fluoride", "acid"],  # Hydrogen Fluoride / Hydrofluoric Acid
    "NO": ["oxide"],  # Nitric Oxide
    "CO": ["monoxide", "oxide"],  # Carbon Monoxide
}


def is_smiles(token, previous_word=None, USE_LLM=False, context=None, model=None):
    """
    It determines whether a given token is a valid SMILES string based on regex rules and optional LLM classification.

    Args:
        token (str): The token to evaluate.
        previous_word (str, optional): The word preceding the token in the text, used for context.
        USE_LLM (bool, optional): Whether to use an LLM for additional validation.
        context (str, optional): Surrounding text context for LLM evaluation.
        model (object, optional): An LLM model instance with a generate_content method.

    Returns:
        bool: True if the token is a valid SMILES string, False otherwise.
    """
    clean_token = token.strip(".,;:!?\"'")
    if not clean_token:
        return False
    if clean_token in EXCLUDE_WORDS:
        return False

    if FALSE_POSITIVE_PATTERN.match(clean_token):
        return False

    # Exclude patterns like ".1", "C.2", "B.3" (could be decimal numbers or codes)
    if re.search(r"\.\d", clean_token):
        return False

    # Exclude patterns like SS1, SS2.1, SC5.1 (Sections/Codes)
    if re.match(r"^(SS|SC)\d+(\.\d+)*$", clean_token):
        return False

    # Exclude Roman Numerals (II, III, IV, VI, etc.)
    if re.fullmatch(r"\(?(?:I{1,3}|IV|VI{0,3}|IX)\)?", clean_token):
        return False

    # they are not smiles
    if clean_token.endswith(("(s)", "(l)", "(g)", "(aq)", ".s")):
        return False

    # If there is a  '/' or '\', there MUST be a double bond '='.
    if ("/" in clean_token or "\\" in clean_token) and "=" not in clean_token:
        return False

    # Out of the brackets, numbers are ring start and end
    token_no_brackets = re.sub(r"\[.*?\]", "", clean_token)
    digits_outside = [c for c in token_no_brackets if c.isdigit()]

    if digits_outside:
        from collections import Counter

        digit_counts = Counter(digits_outside)
        for d, count in digit_counts.items():
            if count % 2 != 0:
                return False

    # Rule: Must contain at least one capital letter OR one number
    if not (re.search(r"[A-Z]", clean_token) or re.search(r"\d", clean_token)):
        return False

    # SMILES cannot start with a number
    if clean_token[0].isdigit():
        return False

    if USE_LLM and model and len(clean_token) <= 6:
        try:
            time.sleep(1.5)
            prompt = (
                f"Role: You are a strict chemical entity classifier.\n"
                f"Task: Determine if the 'Target Token' below is a valid SMILES string or Chemical Formula acting as a molecule identifier in the given context.\n\n"
                f'Context: "...{context}..."\n'
                f'Target Token: "{clean_token}"\n\n'
                f"Rules:\n"
                f"1. YES if it is a SMILES string (e.g., 'c1ccccc1', 'C(=O)O') or a specific chemical formula (e.g., 'H2SO4', 'CH4') used to denote a substance.\n"
                f"2. NO if it is an English word (e.g., 'At', 'Is', 'No', 'Us').\n"
                f"3. NO if it is a non-chemical abbreviation (e.g., 'IF' for Intermediate Frequency, 'IT' for Information Technology).\n"
                f"4. NO if it is a mathematical variable, unit, or section label (e.g., 'V', 'C', '1a').\n"
                f"5. NO if it is a Roman numeral (e.g., 'IV', 'VI').\n\n"
                f"Question: Is the Target Token exclusively representing a chemical structure or formula in this context?\n"
                f"Answer (strictly 'YES' or 'NO'):"
            )
            response = model.generate_content(prompt)
            if response and response.text:
                answer = response.text.strip().upper()
                if "YES" in answer:
                    return True
                if "NO" in answer:
                    return False
                # If answer is unclear, fall through to standard rules
        except Exception:
            # If LLM fails, silently fall back to regex rules
            pass

    has_dash_num = re.search(r"-\d", clean_token)
    if has_dash_num:
        # if it contains a dash followed by a number, check further conditions
        is_ion = re.search(r"\[[^\]]*-\d[^\]]*\]", clean_token)

        # if it contains a pattern like "digit-digit", it is likely a ring bond
        is_ring_bond = re.search(r"\d-\d", clean_token)

        # if it is longer than 8 characters, it is likely a complex molecule
        is_complex = len(clean_token) > 8

        if not is_ion and not is_ring_bond and not is_complex:
            return False

    # Exclude patterns like "-Abc", which are likely not SMILES
    if re.search(r"-[A-Z][a-z]{2,}", clean_token):
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
    # It is valid only if the previous word corresponds to the element name (e.g. "Sodium")
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
        # Check if token is a special char, digit, or bracket (anything that is not an element or aromatic letter)
        if re.match(P_SPECIAL, t) or re.match(P_BRACKET, t):
            has_special_or_digit = True

    if has_aromatic and not has_special_or_digit:
        return False

    return True


def annotate_smiles(text, USE_LLM=False, model=None):
    """
    Annotates SMILES strings in the input text by wrapping them with [START_SMILES] and [END_SMILES] tags.

    Args:
        text (str): The input text to annotate.
        USE_LLM (bool, optional): Whether to use an LLM for additional validation.
        model (object, optional): An LLM model instance with a generate_content method.

    Returns:
        str: The annotated text with SMILES strings tagged.
        int: The count of SMILES strings annotated.
    """
    tokens = text.split()
    new_tokens = []
    smiles_count = 0
    for i, token in enumerate(tokens):
        # If already annotated, keep as is
        if "[START_SMILES]" in token and "[END_SMILES]" in token:
            new_tokens.append(token)
            continue

        previous_word = tokens[i - 1] if i > 0 else None
        context = None
        if USE_LLM:
            start_index = max(0, i - 7)
            end_index = min(len(tokens), i + 3)
            context_tokens = tokens[start_index:end_index]
            context = " ".join(context_tokens)
        # Check if the token (or stripped version) is a SMILES
        # "C[C@]12...," -> "[START_SMILES]C[C@]12...[END_SMILES],"
        match = re.match(r"^([^\w\s\[\]]*)(.*?)([^\w\s\[\]]*)$", token)
        if match:
            prefix, core, suffix = match.groups()
            if is_smiles(core, previous_word, USE_LLM, context, model):
                new_tokens.append(f"{prefix}[START_SMILES]{core}[END_SMILES]{suffix}")
                smiles_count += 1
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)

    return " ".join(new_tokens), smiles_count
