
import json
import time
import os
import csv

if __name__ == "__main__":
    en_corpus_train = open(os.getcwd() + "/../data/train/en.txt", "w")
    ko_corpus_train = open(os.getcwd() + "/../data/train/ko.txt", "w")

    train_json = open(os.getcwd() + "/../data/raw/json/keat-nia/development.json", 'r')
    train_articles = json.loads(train_json.read())

    for article in train_articles:
        sentences = article["text"]
        for sentence in sentences:
            en_translation = sentence["en_text"]
            ko_translation = sentence["ko_text"]

            en_corpus_train.write(en_translation + '\n')
            ko_corpus_train.write(ko_translation + '\n')

    train_json.close()

    en_texts = []
    ko_texts = []

    text_folders = os.getcwd() + "/../data/raw/text"
    for text_folder in os.listdir(text_folders):
        text_folder = text_folders + '/' + text_folder
        for text_file in os.listdir(text_folder):
            text_file = text_folder + '/' + text_file
            if ".en" in text_file:
                en_texts.append(text_file)
            elif ".ko" in text_file:
                ko_texts.append(text_file)

    for en_text, ko_text in zip(sorted(en_texts), sorted(ko_texts)):
        en_text = open(en_text, 'r')
        ko_text = open(ko_text, 'r')

        for en_sentence, ko_sentence in zip(en_text, ko_text):
            en_corpus_train.write(en_sentence)
            ko_corpus_train.write(ko_sentence)

        en_text.close()
        ko_text.close()

    en_ko_csv = open(os.getcwd() + "/../data/raw/csv/keat-nia/corpus.csv", 'r')
    en_ko_csv.readline()

    records = csv.reader(en_ko_csv)
    for ko_sentence, en_sentence in records:
        ko_corpus_train.write(ko_sentence.rstrip() + '\n')
        en_corpus_train.write(en_sentence.rstrip() + '\n')
    en_ko_csv.close()

    en_corpus_train.close()
    ko_corpus_train.close()

