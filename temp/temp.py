import re

input_file = "text_with_smiles.txt"
output_file = "text_with_smiles.txt"

def extract_context(line):
    words = re.findall(r'\b\w+\b|\[START_SMILES\]', line)
    contexts = []
    for i, word in enumerate(words):
        if word == '[START_SMILES]':
            before = ' '.join(words[max(0, i-5):i])
            after = ' '.join(words[i+1:i+6])
            context = f"{before} [START_SMILES] {after}"
            contexts.append(context.strip())
    return contexts

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

result = []
for line in lines:
    contexts = extract_context(line)
    result.extend(contexts)

with open(output_file, "w", encoding="utf-8") as f:
    for context in result:
        f.write(context + "\n")