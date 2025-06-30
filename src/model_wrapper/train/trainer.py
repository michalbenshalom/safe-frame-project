from torch import nn, optim
from tqdm import tqdm

def train(model, train_loader, val_loader, config, model_name):
    """
    Function to handle the training process.
    """

    device = config.get("device", "cpu")
    epochs = config.get("epochs", 5)
    learning_rate = config.get("learning_rate", 2e-5)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training process for {model_name}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            print(type(batch), len(batch), batch)
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if inputs.ndim == 3:
                inputs = inputs.unsqueeze(0)
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
            labels = labels.long()
            outputs = model(inputs).logits if hasattr(model(inputs), 'logits') else model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    print("Training complete.")