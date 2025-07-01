import copy  # מאפשר לשכפל אובייקטים (לשמירת מצב המודל הטוב ביותר)
from datetime import datetime  # ליצירת חותמת זמן לשם הקובץ
import os  # לעבודה עם משתני סביבה ונתיבים
from torch import nn, optim  # מודולים לאימון רשתות נוירונים
import torch  # ספריית PyTorch
from tqdm import tqdm  # פסי התקדמות בלולאות
import io  # עבודה עם buffer בזיכרון
import boto3  # עבודה עם AWS S3
from src.config import MODEL_TYPE  # סוג המודל (לשם הקובץ)
from dotenv import load_dotenv  # טעינת משתני סביבה מקובץ .env

from src.data_management.s3_manager import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET_NAME  # פרטי AWS

load_dotenv()  # טען משתני סביבה מהקובץ .env

# קרא את משתני הסביבה (אם לא נטענו כבר)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

def train(model, train_loader, val_loader, config):
    """
    מאמן את המודל, שומר את המודל הטוב ביותר ישירות ל-S3 (ללא קובץ פיזי).
    :param model: מודל PyTorch
    :param train_loader: DataLoader לאימון
    :param val_loader: DataLoader לוולידציה
    :param config: קונפיגורציה (מילון)
    :param model_name: שם המודל (לשימוש עתידי)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config.get("epochs", 5)  # מספר אפוקים
    lr = config.get("learning_rate", 2e-5)  # קצב למידה
    patience = config.get("early_stopping_patience", 3)  # סבלנות ל-early stopping
    # נתיב שמירה ב-S3
    filename = generate_model_filename()
    s3_path = f"Models/{MODEL_TYPE}/{filename}"

    model.to(device)  # העבר את המודל למכשיר המתאים
    criterion = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()  #nn.CrossEntropyLoss() # פונקציית הפסד לסיווג בינארי
    optimizer = optim.Adam(model.parameters(), lr=lr)  # אופטימייזר

    best_model_state, best_val_loss = None, float('inf')  # שמור את המודל הטוב ביותר
    epochs_without_improvement = 0  # מונה לאפוקים ללא שיפור

    for epoch in range(epochs):
        # אפוק אימון
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs)
        # אפוק ולידציה
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, epochs)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

        improved = val_loss < best_val_loss  # האם יש שיפור בוולידציה
        if improved:
            try:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())  # שמור את מצב המודל
                # שמור ישירות ל-S3 (ללא קובץ פיזי)
                buffer = io.BytesIO()
                torch.save(best_model_state, buffer)
                buffer.seek(0)
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=AWS_ACCESS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY
                )
                s3.upload_fileobj(buffer, AWS_BUCKET_NAME, s3_path)
                print(f"Saved best model so far to s3://{AWS_BUCKET_NAME}/{s3_path} (Val Loss={val_loss:.4f})")
                epochs_without_improvement = 0  # אפס את מונה הסבלנות
            except Exception as e:
                print(f"Failed to save model to S3: {e}")
        else:
            epochs_without_improvement += 1  # הגדל את מונה הסבלנות
            print(f"No improvement. Patience: {epochs_without_improvement}/{patience}")
            if epochs_without_improvement >= patience:
                print("Early stopping.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)  # טען את המודל הטוב ביותר
        print(f"Loaded best model (Val Loss={best_val_loss:.4f})")


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    """
    מבצע אפוק אחד של אימון
    :return: ממוצע הפסד ודיוק
    """
    model.train()  # העבר למצב אימון
    total_loss, correct, total = 0.0, 0, 0  # מונים

    for inputs, labels in tqdm(dataloader, desc=f"[Train] Epoch {epoch+1}/{total_epochs}"):
        inputs, labels = preprocess(inputs, labels, device)  # קדם עיבוד

        optimizer.zero_grad()  # אפס גרדיאנטים
        outputs = forward_pass(model, inputs)  # חישוב קדימה
        loss = criterion(outputs, labels)  # חישוב הפסד
        loss.backward()  # חישוב גרדיאנטים
        optimizer.step()  # עדכון משקלים

        total_loss += loss.item()  # צבירת הפסד
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)  # צבירת דוגמאות

    avg_loss = total_loss / len(dataloader)  # ממוצע הפסד
    accuracy = correct / total * 100  # אחוז דיוק
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device, epoch, total_epochs):
    """
    מבצע אפוק אחד של ולידציה (ללא עדכון משקלים)
    :return: ממוצע הפסד ודיוק
    """
    model.eval()  # מצב הערכה
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():  # ללא גרדיאנטים
        for inputs, labels in tqdm(dataloader, desc=f"[Val] Epoch {epoch+1}/{total_epochs}"):
            inputs, labels = preprocess(inputs, labels, device)  # קדם עיבוד
            outputs = forward_pass(model, inputs)  # חישוב קדימה
            loss = criterion(outputs, labels)  # חישוב הפסד

            total_loss += loss.item()  # צבירת הפסד
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)  # צבירת דוגמאות

    avg_loss = total_loss / len(dataloader)  # ממוצע הפסד
    accuracy = correct / total * 100  # אחוז דיוק
    return avg_loss, accuracy

def preprocess(inputs, labels, device):
    """
    מבצע קדם-עיבוד לנתונים: העברה למכשיר, התאמת מימדים, המרת טיפוס
    """
    inputs, labels = inputs.to(device), labels.to(device)  # העבר למכשיר

    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)  # הוסף batch dimension אם צריך
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)  # הוסף batch dimension ל-label

    labels = labels.float()  # BCEWithLogitsLoss דורשת float ולא long
    return inputs, labels

def forward_pass(model, inputs):
    """
    מבצע חישוב קדימה (forward) עבור המודל
    :return: logits או פלט המודל
    """
    output = model(inputs)
    return output.logits if hasattr(output, 'logits') else output

def generate_model_filename():
    """
    יוצר שם קובץ ייחודי למודל לפי סוג המודל והזמן
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{MODEL_TYPE}_{timestamp}_best.pt"