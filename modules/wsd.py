"""
Experiment 10: Word Sense Disambiguation (WSD) using LSTM / GRU
Disambiguates polysemous library words (e.g. 'novel' -> BOOK vs NEW/INNOVATIVE)
using a Bidirectional LSTM / GRU neural architecture in TensorFlow / Keras.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wsd_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

class WSDLSTMDisambiguator:
    def __init__(self, rnn_type="lstm", max_vocab=500, max_len=25, embedding_dim=32):
        self.rnn_type = rnn_type.lower()
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.metrics = {}

    def load_dataset(self):
        df = pd.read_csv(DATA_PATH)
        return df

    def build_model(self, num_classes):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense, Dropout

        model = Sequential([
            Embedding(input_dim=self.max_vocab, output_dim=self.embedding_dim),
            Dropout(0.2),
            Bidirectional(LSTM(32)) if self.rnn_type == "lstm" else Bidirectional(GRU(32)),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(num_classes, activation="softmax")
        ])

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        return model

    def train_and_evaluate(self, epochs=25, batch_size=4):
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        df = self.load_dataset()
        sentences = df["sentence"].tolist()
        labels = df["sense"].tolist()

        # 1. Fit Tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_vocab, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(sentences)

        # 2. Text to sequences & padding
        sequences = self.tokenizer.texts_to_sequences(sentences)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")

        # 3. Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)

        # 4. Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # 5. Build and train model
        self.model = self.build_model(num_classes)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # 6. Evaluate
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        self.is_trained = True

        self.training_history = {
            "loss": [float(x) for x in history.history["loss"]],
            "val_loss": [float(x) for x in history.history["val_loss"]],
            "accuracy": [float(x) for x in history.history["accuracy"]],
            "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
            "epochs": list(range(1, epochs + 1))
        }

        self.metrics = {
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "final_train_acc": float(history.history["accuracy"][-1]),
            "final_val_acc": float(history.history["val_accuracy"][-1]),
            "classes": list(self.label_encoder.classes_),
            "total_samples": len(df),
            "rnn_type": self.rnn_type.upper()
        }

        # Save model and tokenizer
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_save_path = os.path.join(MODEL_DIR, f"wsd_{self.rnn_type}_model.keras")
        self.model.save(model_save_path)
        
        with open(os.path.join(MODEL_DIR, "wsd_tokenizer.pkl"), "wb") as f:
            pickle.dump(self.tokenizer, f)

        with open(os.path.join(MODEL_DIR, "wsd_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)

        return self.metrics

    def load_saved_model(self):
        import tensorflow as tf
        model_save_path = os.path.join(MODEL_DIR, f"wsd_{self.rnn_type}_model.keras")
        tok_path = os.path.join(MODEL_DIR, "wsd_tokenizer.pkl")
        enc_path = os.path.join(MODEL_DIR, "wsd_encoder.pkl")

        if os.path.exists(model_save_path) and os.path.exists(tok_path) and os.path.exists(enc_path):
            self.model = tf.keras.models.load_model(model_save_path)
            with open(tok_path, "rb") as f:
                self.tokenizer = pickle.load(f)
            with open(enc_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            self.is_trained = True
            return True
        return False

    def disambiguate(self, sentence, target_word="novel"):
        """
        Disambiguates the sense of the target word in the input sentence.
        """
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        if not self.is_trained:
            if not self.load_saved_model():
                self.train_and_evaluate()

        seq = self.tokenizer.texts_to_sequences([sentence])
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")

        preds = self.model.predict(padded, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        predicted_sense = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(preds[pred_idx])

        # Meaning definitions for library context
        definitions = {
            "BOOK": "A book of fiction/literature containing a narrative story or prose.",
            "NEW": "Original, innovative, unprecedented, fresh, or modern approach."
        }

        all_probs = {}
        for cls_name, prob in zip(self.label_encoder.classes_, preds):
            all_probs[cls_name] = round(float(prob), 4)

        return {
            "sentence": sentence,
            "target_word": target_word,
            "predicted_sense": predicted_sense,
            "confidence": round(confidence, 4),
            "meaning_explanation": definitions.get(predicted_sense, "Domain context sense"),
            "probabilities": all_probs,
            "rnn_architecture": f"Bidirectional {self.rnn_type.upper()}"
        }

# Singleton instance
wsd_model = WSDLSTMDisambiguator(rnn_type="lstm")
