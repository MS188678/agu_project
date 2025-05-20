import torch
from torch.utils.data import DataLoader
from transformers import MarianTokenizer, MarianMTModel
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("opus_books", "en-pl")

model_name = "Helsinki-NLP/opus-mt-pl-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def preprocess(example):
    pl, en = example['translation']['pl'], example['translation']['en']
    inputs = tokenizer(pl, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(text_target=en, padding="max_length", truncation=True, max_length=128)
    labels_ids = [id if id != tokenizer.pad_token_id else -100 for id in labels['input_ids']]
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels_ids
    }

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.dataset[idx].items()}

train_dataset = TranslationDataset(tokenized_dataset)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_sentences = [
    "Kot siedzi na macie.",
    "To jest testowe zdanie.",
    "Lubię uczyć się języków obcych."
]
references = [
    ["The cat is sitting on the mat."],
    ["This is a test sentence."],
    ["I like learning foreign languages."]
]

def evaluate_bleu(model):
    model.eval()
    hypotheses = []
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        translated = model.generate(**inputs)
        output = tokenizer.decode(translated[0], skip_special_tokens=True)
        hypotheses.append(output)
    bleu_scores = [
        sentence_bleu([word_tokenize(r)], word_tokenize(h))
        for r, h in zip([r[0] for r in references], hypotheses)
    ]
    return sum(bleu_scores)/len(bleu_scores), hypotheses

def train(model, epochs):
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model

bleu_scores = []
outputs = {}

model.to(device)
b0, o0 = evaluate_bleu(model)
bleu_scores.append(b0)
outputs["0 epok"] = o0

model = train(model, 10)
b10, o10 = evaluate_bleu(model)
bleu_scores.append(b10)
outputs["10 epok"] = o10

model = train(model, 10)
b20, o20 = evaluate_bleu(model)
bleu_scores.append(b20)
outputs["20 epok"] = o20

plt.plot(["0 epok", "10 epok", "20 epok"], bleu_scores, marker='o')
plt.title("BLEU score vs liczba epok treningu")
plt.xlabel("Liczba epok")
plt.ylabel("Średni BLEU score")
plt.grid(True)
plt.show()

for step, out in outputs.items():
    print(f"\n===> Tłumaczenia po {step}:")
    for src, hyp in zip(test_sentences, out):
        print(f"PL: {src}\nEN: {hyp}\n")
