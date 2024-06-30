import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier:
    def __init__(self, model_name, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.model = self.model.to(self.device)

    def train(self, train_data, val_data, epochs, batch_size, learning_rate, max_len):
        train_dataset = SentimentDataset(train_data['texts'], train_data['labels'], self.tokenizer, max_len)
        val_dataset = SentimentDataset(val_data['texts'], val_data['labels'], self.tokenizer, max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, correct_bias=False)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            print(f'Train loss: {train_loss / len(train_loader)}')

            self.model.eval()
            eval_loss = 0
            eval_accuracy = 0
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                eval_loss += loss.item()
                preds = torch.argmax(logits, dim=1).flatten()
                accuracy = (preds == labels).cpu().numpy().mean() * 100
                eval_accuracy += accuracy
            print(f'Validation loss: {eval_loss / len(val_loader)}')
            print(f'Validation accuracy: {eval_accuracy / len(val_loader)}')

    def predict(self, text, max_len):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1).item(), probs

if __name__ == "__main__":
    # Sample data
    texts = ["I love this!", "I hate this!", "This is okay.", "I am so happy.", "I am very sad."]
    labels = [1, 0, 1, 1, 0]  # 1 for positive, 0 for negative

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_data = {'texts': train_texts, 'labels': train_labels}
    val_data = {'texts': val_texts, 'labels': val_labels}

    model_name = 'bert-base-uncased'
    num_classes = 2
    epochs = 3
    batch_size = 2
    learning_rate = 2e-5
    max_len = 64

    classifier = SentimentClassifier(model_name, num_classes)
    classifier.train(train_data, val_data, epochs, batch_size, learning_rate, max_len)

    # Prediction
    test_text = "I am extremely happy!"
    label, probs = classifier.predict(test_text, max_len)
    print(f'Predicted label: {label}, Probabilities: {probs}')
