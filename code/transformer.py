
import numpy as np
import tensorflow as tf

SEQUENCE_LEN = 160
EMBEDDING_DIM = 1024
NUM_HEADS = 12
QUERY_KEY_DIM = 64
VALUE_DIM = 64
DROPOUT_RATE = 0.4
UPSAMPLE_DIM = 4096
N_ENCODER = 6
N_DECODER = 6

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

