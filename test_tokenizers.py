from custom_tokenizers import (
    CharacterTokenizer,
    ElementTokenizer,
    ElementAllParenthesisTokenizer,
    ElementAromaticsTokenizer,
    ElementNoParenthesisTokenizer,
    ElementRingsTokenizer,
)

def test_tokenizers(test_formulas):
    tokenizers = {
        "CharacterTokenizer": CharacterTokenizer(),
        "ElementTokenizer": ElementTokenizer(),
        "ElementAllParenthesisTokenizer": ElementAllParenthesisTokenizer(),
        "ElementAromaticsTokenizer": ElementAromaticsTokenizer(),
        "ElementNoParenthesisTokenizer": ElementNoParenthesisTokenizer(),
        "ElementRingsTokenizer": ElementRingsTokenizer(),
    }

    for name, tokenizer in tokenizers.items():
        print(f"Testing {name}:")
        for formula in test_formulas:
            tokens = tokenizer(formula)["input_ids"]
            print(f"  Tokens for {formula}: {tokens}")
            decoded = tokenizer.decode(tokens)
            assert decoded.replace('[UNK]', '') == formula.replace('[UNK]', ''), f"Mismatch in {name} for formula {formula}"

if __name__ == "__main__":
    test_formulas = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "C1=CC=CC=C1",
        "N[C@@H](C)C(=O)O",
        "C[C@H](N)C(=O)O",
        "BrCCCl",
        "C1CCCCC1",
    ]
    test_tokenizers(test_formulas)