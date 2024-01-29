
import time
import os
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

BATCH_SIZE = 8
SHUFFLE_BUFFER_SIZE = 65536
SEQUENCE_LEN = 256
EMBEDDING_DIM = 512
NUM_HEADS = 6
QUERY_KEY_DIM = 64
VALUE_DIM = 64
DROPOUT_RATE = 0.1
UPSAMPLE_DIM = 1024
N_ENCODER = 6
N_DECODER = 6
N_EPOCH = 1
LEARNING_RATE = 1e-8

def sentence_pair_generator(sentences, translations):
    sentences = open(sentences, 'r')
    translations = open(translations, 'r')
    def generator():
        for sentence, translation in zip(sentences, translations):
            yield sentence.rstrip(), translation.rstrip()
        sentences.seek(0)
        translations.seek(0)
    return generator

def tokenize(sentences, translations, sentence_tokenizer, translation_tokenizer):
    sentences = [sentence.decode("utf-8") for sentence in sentences.numpy().tolist()]
    translations = [translation.decode("utf-8") for translation in translations.numpy().tolist()]

    tokenized_sentences = [["[BOS]"] + sentence_tokenizer.tokenize(sentence) + ["[EOS]"] for sentence in sentences]
    tokenized_translations = [["[BOS]"] + translation_tokenizer.tokenize(translation) for translation in translations]
    tokenized_labels = [translation_tokenizer.tokenize(translation) + ["[EOS]"] for translation in translations]

    encoded_sentences = sentence_tokenizer(
        tokenized_sentences, 
        add_special_tokens = False,
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_LEN,
        is_split_into_words = True,
        return_tensors = "tf"
        )['input_ids']

    encoded_translations = translation_tokenizer(
        tokenized_translations, 
        add_special_tokens = False,
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_LEN,
        is_split_into_words = True,
        return_tensors = "tf"
        )['input_ids']

    encoded_labels = translation_tokenizer(
        tokenized_labels, 
        add_special_tokens = False,
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_LEN,
        is_split_into_words = True,
        return_tensors = "tf"
        )['input_ids']

    return encoded_sentences, encoded_translations, encoded_labels

class FeedForward(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.nonlinear = tf.keras.Sequential([
            tf.keras.layers.Dense(UPSAMPLE_DIM, activation="relu"),
            tf.keras.layers.Dense(EMBEDDING_DIM),  
            tf.keras.layers.Dropout(DROPOUT_RATE)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_normalization = tf.keras.layers.LayerNormalization()

    def call(self, embedding):
        nonlinear = self.nonlinear(embedding)
        residual = self.add([embedding, nonlinear])
        normalized = self.layer_normalization(residual)
        return normalized

class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads = NUM_HEADS,
            key_dim = QUERY_KEY_DIM, 
            value_dim = VALUE_DIM,
            )
        self.add = tf.keras.layers.Add()
        self.layer_normalization = tf.keras.layers.LayerNormalization() 

    def call(self, embedding):
        attention = self.attention(
            query = embedding,
            key = embedding,
            value = embedding
            )
        residual = self.add([attention, embedding])
        normalized = self.layer_normalization(residual)
        return normalized 

class MaskedAttention(Attention):
    def __init__(self):
        super().__init__()

    def call(self, embedding):
        attention = self.attention(
            query = embedding,
            key = embedding,
            value = embedding,
            use_causal_mask = True
            )
        residual = self.add([attention, embedding])
        normalized = self.layer_normalization(residual)
        return normalized

class ContextualAttention(Attention):
    def __init__(self):
        super().__init__()

    def call(self, embedding, context):
        attention = self.attention(
            query = embedding,
            key = context,
            value = context
            )
        residual = self.add([attention, embedding])
        normalized = self.layer_normalization(residual)
        return normalized

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.feed_forward = FeedForward()

    def call(self, embedding):
        attention = self.attention(embedding)
        nonlinear = self.feed_forward(attention)
        return nonlinear

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.masked_attention = MaskedAttention()
        self.contextual_attention = ContextualAttention()
        self.feed_forward = FeedForward()

    def call(self, embedding, context):
        masked_attention = self.masked_attention(embedding)
        contextual_attention = self.contextual_attention(masked_attention, context)
        nonlinear = self.feed_forward(contextual_attention)
        return nonlinear

class Embedder(tf.keras.layers.Layer):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedder = tf.keras.layers.Embedding(
            input_dim = vocab_size,
            output_dim = EMBEDDING_DIM,
            input_length = SEQUENCE_LEN,
            mask_zero = True
            )

    def positional_encoding(self):
        positions = np.arange(SEQUENCE_LEN)[:, np.newaxis] 
        rates = 1 / (10000 ** (np.arange(0, EMBEDDING_DIM, 2)[np.newaxis, :] / EMBEDDING_DIM))
        radians = positions * rates
        sines = np.sin(radians)
        cosines = np.cos(radians)
        encoding = np.concatenate([sines, cosines], axis=-1)
        return tf.cast(encoding, dtype=tf.float32)[tf.newaxis, :SEQUENCE_LEN, :]

    def compute_mask(self, *args, **kwargs):
        return self.embedder.compute_mask(*args, **kwargs)

    def call(self, encoding):
        embedding = self.embedder(encoding) + self.positional_encoding()
        return embedding 


class Transformer(tf.keras.Model):
    def __init__(self, sentence_vocab_size, translation_vocab_size):
        super().__init__()
        self.embed_sentence = Embedder(sentence_vocab_size)
        self.embed_translation = Embedder(translation_vocab_size)
        self.encoders = [Encoder() for _ in range(N_ENCODER)]
        self.decoders = [Decoder() for _ in range(N_DECODER)]
        self.logits = tf.keras.layers.Dense(translation_vocab_size)

    def call(self, encoded_sentence_pair):
        encoded_sentence, encoded_translation = encoded_sentence_pair

        embedded_sentence = self.embed_sentence(encoded_sentence)
        embedded_translation = self.embed_translation(encoded_translation)

        for encoder in self.encoders:
            embedded_sentence = encoder(embedded_sentence)

        for decoder in self.decoders:
            embedded_translation = decoder(embedded_translation, embedded_sentence)

        return self.logits(embedded_translation)

if __name__ == "__main__":
    sentence_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    translation_tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")

    sentence_tokenizer.add_special_tokens({"eos_token": "[EOS]", "bos_token": "[BOS]"})
    translation_tokenizer.add_special_tokens({"eos_token": "[EOS]", "bos_token": "[BOS]"})

    train_sentence_pairs = tf.data.Dataset.from_generator(
        generator = sentence_pair_generator(
            os.getcwd() + "/data/train/en.txt", 
            os.getcwd() + "/data/train/ko.txt"
        ), 
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = tf.data.AUTOTUNE).repeat(N_EPOCH)

    valid_sentence_pairs = tf.data.Dataset.from_generator(
        generator = sentence_pair_generator(
            os.getcwd() + "/data/valid/en.txt", 
            os.getcwd() + "/data/valid/ko.txt"
        ), 
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )).batch(BATCH_SIZE).prefetch(buffer_size = tf.data.AUTOTUNE)

    model = Transformer(sentence_tokenizer.vocab_size + 2, translation_tokenizer.vocab_size + 2)
    error = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = LEARNING_RATE)

    for sentences, translations in train_sentence_pairs:
        encoded_sentences, encoded_translations, encoded_labels = tokenize(sentences, translations, sentence_tokenizer, translation_tokenizer)
        with tf.GradientTape() as tape:
            predicted_logits = model((encoded_sentences, encoded_translations), training = True)
            prediction_error = error(encoded_labels, predicted_logits)
        gradients = tape.gradient(prediction_error, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # change to subset of valid set
    for sentences, translations in valid_sentence_pairs:
        encoded_sentences, encoded_translations, encoded_labels = tokenize(sentences, translations, sentence_tokenizer, translation_tokenizer)
        predicted_logits = model((encoded_sentences, encoded_translations), training = False)
        predicted_probs = tf.nn.softmax(predicted_logits, axis=-1)
        predicted_token_sequences = tf.argmax(predicted_probs, axis=-1).numpy().tolist()

        for sentence, predicted_token_sequence, translation in zip(sentences, predicted_token_sequences, translations):
            print(f"sentence:", sentence.numpy().decode("utf-8"), flush=True)
            print("predicted translation:", translation_tokenizer.decode(predicted_token_sequence, skip_special_tokens=True), flush=True)
            print("actual translation:", translation.numpy().decode("utf-8"), '\n', flush=True)
            time.sleep(60)
