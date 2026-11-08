import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. تحميل البيانات
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# 2. إعداد BERT Tokenizer والنموذج
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. تحويل التصنيفات وتقسيم البيانات
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['message'].tolist(), 
    df['label_num'].tolist(), 
    test_size=0.2, 
    random_state=42
)

print("تم تجهيز كود BERT بنجاح وجاهز للتدريب!")
