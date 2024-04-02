
import os
import re
import sys
import json
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
from transformer import *

PER_REPLICA_BATCH = 256
N_EPOCH = 512
LEARNING_RATE = 2e-6

EN_VOCAB = 30524
KO_VOCAB = 42002
EN_BOS = 30522
EN_EOS = 30523
KO_BOS = 42000
KO_EOS = 42001

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {"worker": [
                            "c17.cluster:24900",
                            "c18.cluster:24900",
                            "c19.cluster:24900",
                            "c20.cluster:24900"
                           ]
                },
    "task": {"type": "worker", "index": os.environ["SLURM_NODEID"]}
})

def beam_search(sentence, beam_width, model, sentence_tokenizer, translation_tokenizer):
    encoder_in = sentence_tokenizer(sentence, add_special_tokens=False)["input_ids"]
    encoder_in = np.pad(encoder_in, (1, 1), constant_values=(KO_BOS, KO_EOS))
    encoder_in = np.pad(encoder_in, (0, SEQUENCE_LEN - len(encoder_in)), constant_values=0)
    encoder_in = tf.cast(encoder_in[np.newaxis, :], dtype=tf.int32)

    top_k_scores = [0]
    top_k_sequences = [[EN_BOS]]
    top_k_norm_scores = [0]

    for timestep in range(1, SEQUENCE_LEN):
        candidate_scores = []
        candidate_sequences = []
        candidate_norm_scores = []

        for prefix_score, prefix_sequence, prefix_norm_score in zip(top_k_scores, top_k_sequences, top_k_norm_scores):

            if prefix_sequence[-1] == EN_EOS:
                candidate_scores.append(prefix_score)
                candidate_sequences.append(prefix_sequence)
                candidate_norm_scores.append(prefix_norm_score)
                continue

            decoder_in = np.pad(prefix_sequence, (0, SEQUENCE_LEN - len(prefix_sequence)), constant_values=0)
            decoder_in = tf.cast(decoder_in[np.newaxis, :], dtype=tf.int32)

            logits = model((encoder_in, decoder_in), training=False)
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0][timestep-1]

            for token, prob in enumerate(probs):
                candidate_score = prefix_score + np.log(prob)
                candidate_sequence = prefix_sequence + [token]
                candidate_norm_score = candidate_score / len(candidate_sequence)

                candidate_scores.append(candidate_score)
                candidate_sequences.append(candidate_sequence)
                candidate_norm_scores.append(candidate_norm_score)

        top_k_scores = []
        top_k_sequences = []
        top_k_norm_scores = []
        top_k_indices = tf.math.top_k(candidate_norm_scores, k=beam_width).indices.numpy()

        for index in top_k_indices:
            top_k_scores.append(candidate_scores[index])
            top_k_sequences.append(candidate_sequences[index])
            top_k_norm_scores.append(candidate_norm_scores[index])

    most_probable_sequence = top_k_sequences[top_k_norm_scores.index(max(top_k_norm_scores))]
    return translation_tokenizer.decode(most_probable_sequence, skip_special_tokens=True)

def sentence_pair_generator(sentence_file, translation_file, sentence_tokenizer, translation_tokenizer):
    sentence_file = open(sentence_file, 'r')
    translation_file = open(translation_file, 'r')

    sentences = []
    translations = []

    for sentence, translation in zip(sentence_file, translation_file):
        sentences.append(sentence)
        translations.append(translation)

    sentence_file.close()
    translation_file.close()

    encoded_sentences = sentence_tokenizer(sentences, add_special_tokens=False)["input_ids"]
    encoded_sentences = [np.pad(sentence, (1, 1), constant_values=(KO_BOS, KO_EOS)) for sentence in encoded_sentences]
    for sentence in encoded_sentences:
        sentence.resize((SEQUENCE_LEN), refcheck=False)

    translation_encodings = translation_tokenizer(translations, add_special_tokens=False)["input_ids"]

    encoded_translations = [np.pad(translation, (1, 0), constant_values=EN_BOS) for translation in translation_encodings]
    for translation in encoded_translations:
        translation.resize((SEQUENCE_LEN), refcheck=False)

    encoded_labels = [np.pad(translation, (0, 1), constant_values=EN_EOS) for translation in translation_encodings]
    for label in encoded_labels:
        label.resize((SEQUENCE_LEN), refcheck=False)

    def generator():
        for encoded_sentence, encoded_translation, encoded_label in zip(encoded_sentences, encoded_translations, encoded_labels): 
            yield encoded_sentence, encoded_translation, encoded_label

    return generator

def dataset_constructor(sentence_tokenizer, translation_tokenizer):
    def construct_dataset(context):
        dataset = tf.data.Dataset.from_generator(
            generator = sentence_pair_generator(
                os.getcwd() + "/data/train/ko.txt",
                os.getcwd() + "/data/train/en.txt",
                sentence_tokenizer,
                translation_tokenizer
            ), 
            output_signature = (
                tf.TensorSpec(shape=(SEQUENCE_LEN), dtype=tf.int32),
                tf.TensorSpec(shape=(SEQUENCE_LEN), dtype=tf.int32),
                tf.TensorSpec(shape=(SEQUENCE_LEN), dtype=tf.int32)
            )
        ).shard(context.num_input_pipelines, context.input_pipeline_id).cache()
        dataset = dataset.shuffle(dataset.cardinality()).batch(PER_REPLICA_BATCH, drop_remainder=True)
        return dataset.prefetch(tf.data.AUTOTUNE)
    return construct_dataset

nccl = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
)
strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=nccl)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

with strategy.scope():
    model = Transformer(KO_VOCAB, EN_VOCAB)
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE, 
        decay_rate=0.96,
        decay_steps=4096
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=tf.Variable(0))
    manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=os.getcwd() + "/checkpoint", max_to_keep=1)

def train(batch):
    sentences, translations, labels = batch
    with tf.GradientTape() as tape:
        predicted_logits = model((sentences, translations), training=True)
        prediction_loss = loss(labels, predicted_logits)

        mask = tf.cast(labels != 0, prediction_loss.dtype)
        masked_loss = prediction_loss * mask
        reduced_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

    gradients = tape.gradient(reduced_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return reduced_loss

@tf.function
def distributed_train(distributed_batch):
    per_replica_losses = strategy.run(train, args=(distributed_batch,))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

if __name__ == "__main__":

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        if os.environ["SLURM_NODEID"] == '0':
            print(f"LOADED CHECKPOINT @ EPOCH {checkpoint.epoch.value()}", flush=True)

    sentence_tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")
    translation_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sentence_tokenizer.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"})
    translation_tokenizer.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"})

    distributed_dataset = strategy.distribute_datasets_from_function(dataset_constructor(sentence_tokenizer, translation_tokenizer))

    for epoch in range(checkpoint.epoch.value(), N_EPOCH):
        for distributed_batch in distributed_dataset:
            distributed_loss = distributed_train(distributed_batch)

        if os.environ["SLURM_NODEID"] == '0':
            print(f"EPOCH {epoch} COMPLETED @", datetime.now().strftime("%I:%M %p"), flush=True)

            sentence = "안녕하세요 제 이름은 최다윗 입니다."
            print("SENTENCE:", sentence, flush=True)
            print("TRANSLATION:", beam_search(sentence, 6, model, sentence_tokenizer, translation_tokenizer), flush=True)

            manager.save()
            model.save(os.getcwd() + "/model/latest_version.keras")
            print(f"SAVED CHECKPOINT @ EPOCH {epoch}", file=sys.stdout, flush=True)
        checkpoint.epoch.assign_add(1)

