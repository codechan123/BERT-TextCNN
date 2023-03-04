
import json
import pandas as pd

from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report

from BERT_model import build_bert_model
from data_helper import load_data
from bert4keras.optimizers import Adam

# 定义超参数

maxlen = 128
batch_size = 32
class_nums = 13

config_path = 'D:/Code/2022/python100day/KBQA-medical/BERT-TextCNN/bert_weight_files/bert_config.json'
checkpoint_path = 'D://Code/2022/python100day/KBQA-medical/BERT-TextCNN/bert_weight_files/bert_model.ckpt'
dict_path = 'D://Code/2022/python100day/KBQA-medical/BERT-TextCNN/bert_weight_files/vocab.txt'


tokenizer = Tokenizer(dict_path)


class data_generator(DataGenerator):
    """数据生成器"""
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

if __name__ == '__main__':

    train_data = load_data('data/train.csv')
    test_data = load_data('data/test.csv')

    train_generator = data_generator(train_data, batch_size)
    test_generator = data_generator(test_data, batch_size)

    model = build_bert_model(config_path, checkpoint_path, class_nums=class_nums)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(5e-6),
        metrics=['accuracy'],
    )

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_acc',
        patience=2,
        verbose=2,
        mode='max'
    )
    best_model_filepath = 'bert_weight.weights'
    checkpoint = keras.callbacks.ModelCheckpoint(
        best_model_filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=test_generator.forfit(),
        validation_steps=len(test_generator),
        shuffle=True,
        callbacks=[earlystop, checkpoint]
    )

    model.load_weights('best_model_weights')
    test_pred = []
    test_true = []
    for x, y in test_generator:
        p = model.predict(x).argmax(axis=1)
        test_pred.extend(p)

    test_true = test_data[:, 1].tolist()
    print(set(test_true))
    print(set(test_true))

    targets_names = [line.strip() for line in open('label', 'r', encoding='utf8')]
    print(classification_report(test_true, test_pred, targets_names=targets_names))


