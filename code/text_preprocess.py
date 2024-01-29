


import json
import time
import os

if __name__ == "__main__":
    en_corpus_train = open(os.getcwd() + "/../data/train/en.txt", "a")
    ko_corpus_train = open(os.getcwd() + "/../data/train/ko.txt", "a")

    en_corpus_valid = open(os.getcwd() + "/../data/valid/en.txt", "a")
    ko_corpus_valid = open(os.getcwd() + "/../data/valid/ko.txt", "a")

    train_json = open(os.getcwd() + "/../data/raw/json/keat-nia/development.json", 'r')
    valid_json = open(os.getcwd() + "/../data/raw/json/keat-nia/evaluation.json", 'r')

    train_articles = json.loads(train_json.read())
    valid_articles = json.loads(valid_json.read())

    train_json.close()
    valid_json.close()

    for article in train_articles:
        sentences = article["text"]
        for sentence in sentences:
            en_translation = sentence["en_text"]
            ko_translation = sentence["ko_text"]

            en_corpus_train.write(en_translation + '\n')
            ko_corpus_train.write(ko_translation + '\n')

    for article in valid_articles:
        sentences = article["text"]
        for sentence in sentences:
            en_translation = sentence["en_text"]
            ko_translation = sentence["ko_text"]

            en_corpus_valid.write(en_translation + '\n')
            ko_corpus_valid.write(ko_translation + '\n')

    en_corpus_valid.close()
    ko_corpus_valid.close()

    en_files = []
    ko_files = []

    text_folders = os.getcwd() + "/../data/raw/text"
    for text_folder in os.listdir(text_folders):
        text_folder = text_folders + '/' + text_folder
        for text_file in os.listdir(text_folder):
            text_file = text_folder + '/' + text_file
            if ".en" in text_file:
                en_files.append(text_file)
            elif ".ko" in text_file:
                ko_files.append(text_file)

    for en_file, ko_file in zip(sorted(en_files), sorted(ko_files)):
        en_file = open(en_file, 'r')
        ko_file = open(ko_file, 'r')

        for en_sentence, ko_sentence in zip(en_file, ko_file):
            en_corpus_train.write(en_sentence)
            ko_corpus_train.write(ko_sentence)

        en_file.close()
        ko_file.close()

    en_corpus_train.close()
    ko_corpus_train.close()



