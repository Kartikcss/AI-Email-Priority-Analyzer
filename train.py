import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Bigger dataset (important)
data = [
    ("urgent meeting at 5pm", "high"),
    ("submit report immediately", "high"),
    ("deadline is today", "high"),
    ("client escalation urgent", "high"),
    ("please respond asap", "high"),

    ("let's schedule a meeting next week", "medium"),
    ("please review when you have time", "medium"),
    ("checking on project progress", "medium"),
    ("follow up on previous email", "medium"),
    ("can we discuss tomorrow", "medium"),

    ("newsletter subscription update", "low"),
    ("discount offer just for you", "low"),
    ("social media notification", "low"),
    ("new blog post available", "low"),
    ("weekly digest email", "low"),
]

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("emails.csv", index=False)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved!")
