import sys
import os
import logging
import pprint
import re


# Regex for SMILES validation
# Matches organic subset, common elements, brackets, and special chars
# Excludes common English words that might match 
ELEMENTS = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
            "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br",
            "Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
            "I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm",
            "Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
            "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
            "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]
ELEMENTS.sort(key=len, reverse=True)
AROMATICS = ['b', 'c', 'n', 'o', 'p', 's']
AROMATICS_SET = set(AROMATICS)

# Pattern components
P_BRACKET = r"\[[^\]]+\]"
P_ELEMENTS = "|".join(ELEMENTS)
P_AROMATICS = "|".join(AROMATICS)
P_SPECIAL = r"[\d=\#\-\+\.\\\/@%\(\)]"

# Combined pattern for tokenization (order important)
# Bracket > Element > Aromatic > Special
TOKEN_PATTERN = f"({P_BRACKET}|{P_ELEMENTS}|{P_AROMATICS}|{P_SPECIAL})"

VALIDATION_PATTERN = re.compile(f"^(?:{P_BRACKET}|{P_ELEMENTS}|{P_AROMATICS}|{P_SPECIAL})+$")

EXCLUDE_WORDS = {"I", "In", "On", "No", "So", "Up", "A", "a", "Be", "Am", "As", "At", "Is", "Or", "Go", "Us", "By", "To", "It", "Re", "Pm", "Mt", "UV"}

def is_smiles(token):
    clean_token = token.strip(".,;:!?\"'")
    if not clean_token:
        return False
    if clean_token in EXCLUDE_WORDS:
        return False

    # SMILES cannot start with a number
    if clean_token[0].isdigit():
        return False
    
    if clean_token.endswith('s') and clean_token[:-1].isupper() and clean_token not in ELEMENTS:
        # Only reject if it looks like a plain text acronym (no digits, no bonds, no brackets)
        if not re.search(r'[\d=\#\-\+\(\)\[\]]', clean_token):
            return False

    # Must contain at least one letter (heuristic to avoid matching pure numbers or symbols like "5" or ">>")
    if not re.search(r'[a-zA-Z]', clean_token):
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
    for token in tokens:
        # Check if the token (or stripped version) is a SMILES
        # "C[C@]12...," -> "[START_SMILES]C[C@]12...[END_SMILES]," 
        # Modified regex to NOT strip brackets [] as they are essential for SMILES (e.g. [Na+])
        match = re.match(r"^([^\w\s\[\]]*)(.*?)([^\w\s\[\]]*)$", token)
        if match:
            prefix, core, suffix = match.groups()
            if is_smiles(core):
                new_tokens.append(f"{prefix}[START_SMILES]{core}[END_SMILES]{suffix}")
                smiles_count += 1
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)
            
    return " ".join(new_tokens), smiles_count