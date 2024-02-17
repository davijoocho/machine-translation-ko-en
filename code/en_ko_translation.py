
import os
import re
import sys
import json
import time
import numpy as np
from datetime import datetime
import tensorflow as tf
from transformers import AutoTokenizer

BATCH_SIZE_PER_REPLICA = 80
N_TRAIN_EXAMPLE = 450000
N_VALID_EXAMPLE = 10

SEQUENCE_LEN = 140
EMBEDDING_DIM = 1024
NUM_HEADS = 12
QUERY_KEY_DIM = 64
VALUE_DIM = 64
DROPOUT_RATE = 0.3
UPSAMPLE_DIM = 4096
N_ENCODER = 6
N_DECODER = 6
N_EPOCH = 128
LEARNING_RATE = 4e-5

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {"worker": [
                            "c3.cluster:24900",
                            "c4.cluster:24900",
                            "c5.cluster:24900",
                            "c6.cluster:24900",
                            "c8.cluster:24900",
                            "c11.cluster:24900",
                            "c12.cluster:24900",
                            "c13.cluster:24900"
                           ]
                },
    "task": {"type": "worker", "index": os.environ["SLURM_NODEID"]}
})

class FeedForward(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.nonlinear = tf.keras.Sequential([
            tf.keras.layers.Dense(UPSAMPLE_DIM, activation="relu", dtype=tf.float32),
            tf.keras.layers.Dense(EMBEDDING_DIM, dtype=tf.float32),  
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
            dtype=tf.float32
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
        depth = EMBEDDING_DIM / 2
        positions = np.arange(SEQUENCE_LEN)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        rates = 1 / (10000 ** depths)
        radians = positions * rates
        encoding = np.concatenate([np.sin(radians), np.cos(radians)], axis=-1)
        return tf.cast(encoding, dtype=tf.float32)[np.newaxis, :, :]

    def compute_mask(self, *args, **kwargs):
        return self.embedder.compute_mask(*args, **kwargs)

    def call(self, encoding):
        factor = tf.math.sqrt(tf.cast(EMBEDDING_DIM, dtype=tf.float32))
        embedding = self.embedder(encoding) 
        scaled_embedding = embedding * factor
        return scaled_embedding + self.positional_encoding()

class Transformer(tf.keras.Model):
    def __init__(self, sentence_vocab_size, translation_vocab_size):
        super().__init__()
        self.embed_sentence = Embedder(sentence_vocab_size)
        self.embed_translation = Embedder(translation_vocab_size)
        self.encoders = [Encoder() for _ in range(N_ENCODER)]
        self.decoders = [Decoder() for _ in range(N_DECODER)]
        self.logits = tf.keras.layers.Dense(translation_vocab_size, dtype=tf.float32)

    def call(self, encoded_sentence_pair):
        encoded_sentence, encoded_translation = encoded_sentence_pair
        embedded_sentence = self.embed_sentence(encoded_sentence)
        embedded_translation = self.embed_translation(encoded_translation)

        for encoder in self.encoders:
            embedded_sentence = encoder(embedded_sentence)

        for decoder in self.decoders:
            embedded_translation = decoder(embedded_translation, embedded_sentence)

        return self.logits(embedded_translation)

sentence_tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")
translation_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sentence_tokenizer.add_special_tokens({"eos_token": "[EOS]", "bos_token": "[BOS]"})
translation_tokenizer.add_special_tokens({"eos_token": "[EOS]", "bos_token": "[BOS]"})
EOS = 30522

def tokenize(sentences, translations):
    tokenized_sentences = [["[BOS]"] + sentence_tokenizer.tokenize(sentence) + ["[EOS]"] for sentence in sentences]
    tokenized_translations = [["[BOS]"] + translation_tokenizer.tokenize(translation) for translation in translations]
    tokenized_labels = [translation_tokenizer.tokenize(translation) + ["[EOS]"] for translation in translations]

    encoded_sentences = sentence_tokenizer(
        tokenized_sentences, 
        add_special_tokens = False,
        padding = "max_length",
        truncation = True,
        max_length = SEQUENCE_LEN,
        is_split_into_words = True,
        return_tensors="tf"
    )["input_ids"]

    encoded_translations = translation_tokenizer(
        tokenized_translations, 
        add_special_tokens = False,
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_LEN,
        is_split_into_words = True,
        return_tensors="tf"
    )["input_ids"]

    encoded_labels = translation_tokenizer(
        tokenized_labels, 
        add_special_tokens = False,
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_LEN,
        is_split_into_words = True,
        return_tensors="tf"
    )["input_ids"]

    return encoded_sentences, encoded_translations, encoded_labels

def sentence_pair_generator(sentence_file, translation_file):
    sentence_file = open(sentence_file, 'r')
    translation_file = open(translation_file, 'r')

    sentences = []
    translations = []
    for sentence, translation in zip(sentence_file, translation_file):
        sentences.append(sentence)
        translations.append(translation)

    sentence_file.close()
    translation_file.close()
    
    encoded_sentences, encoded_translations, encoded_labels = tokenize(sentences, translations)

    def generator():
        for encoded_sentence, encoded_translation, encoded_label in zip(encoded_sentences, encoded_translations, encoded_labels): 
            yield encoded_sentence, encoded_translation, encoded_label

    return generator

def distribute_dataset(context):
    dataset = tf.data.Dataset.from_generator(
        generator = sentence_pair_generator(
            os.getcwd() + "/data/train/ko.txt",
            os.getcwd() + "/data/train/en.txt"
        ), 
        output_signature = (
            tf.TensorSpec(shape=(SEQUENCE_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(SEQUENCE_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(SEQUENCE_LEN), dtype=tf.int32)
        )
    ).shuffle(N_TRAIN_EXAMPLE).cache().batch(BATCH_SIZE_PER_REPLICA, drop_remainder=True)

    dataset = dataset.shard(
        context.num_input_pipelines, 
        context.input_pipeline_id
    ).prefetch(tf.data.AUTOTUNE)

    return dataset

backend = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
)
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=backend
)

with strategy.scope():
    distributed_dataset = strategy.distribute_datasets_from_function(distribute_dataset)
    model = Transformer(
        sentence_tokenizer.vocab_size + 2,
        translation_tokenizer.vocab_size + 2
    )
    error = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE, 
        decay_rate=0.96,
        decay_steps=2048
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)

def train(sentences, translations, labels):
    with tf.GradientTape() as tape:
        predicted_logits = model((sentences, translations), training=True)
        prediction_error = error(labels, predicted_logits)
    gradients = tape.gradient(prediction_error, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return prediction_error

if __name__ == "__main__":
    valid = tf.data.Dataset.from_generator(
        generator = sentence_pair_generator(
            os.getcwd() + "/data/valid/ko.txt",
            os.getcwd() + "/data/valid/en.txt"
        ), 
        output_signature = (
            tf.TensorSpec(shape=(SEQUENCE_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(SEQUENCE_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(SEQUENCE_LEN), dtype=tf.int32)
        )
    ).batch(N_VALID_EXAMPLE)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, epoch=tf.Variable(0))
    manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=os.getcwd() + "/model", max_to_keep=1)

    # CHECKPOINT
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"LOADED CHECKPOINT @ EPOCH {checkpoint.epoch.value()}", file=sys.stdout)

    for epoch in range(checkpoint.epoch.value(), N_EPOCH):
        # TRAIN
        for sentences, translations, labels in distributed_dataset:
            per_replica_error = strategy.run(train, args=(sentences, translations, labels))

        # VALIDATION
        if os.environ["SLURM_NODEID"] == '0':
            print(f"EPOCH {epoch} COMPLETED @", datetime.now().strftime("%I:%M %p"), '\n', file=sys.stdout)

            for sentences, translations, labels in valid:
                predicted_logits = model((sentences, translations), training=False)
                predicted_probabilities = tf.nn.softmax(predicted_logits, axis=-1)
                predicted_translations = tf.argmax(predicted_probabilities, axis=-1)

                for sentence, translation in zip(sentences, predicted_translations):
                    translation = translation.numpy()
                    filtered_translation = []
                    idx = 0
                    while idx < len(translation) and translation[idx] != EOS:
                        filtered_translation.append(translation[idx])
                        idx = idx + 1 

                    print("SENTENCE:", 
                           sentence_tokenizer.decode(sentence.numpy(), skip_special_tokens=True),
                           file=sys.stdout,
                           flush=True)

                    print("TRANSLATION:", 
                           translation_tokenizer.decode(filtered_translation, skip_special_tokens=False),
                           '\n',
                           file=sys.stdout, 
                           flush=True)

        # CHECKPOINT
        if checkpoint.epoch % 8 == 0:
            if os.environ["SLURM_NODEID"] == '0':
                manager.save()
                print(f"SAVED CHECKPOINT @ EPOCH {epoch}", file=sys.stdout, flush=True)
        checkpoint.epoch.assign_add(1)

