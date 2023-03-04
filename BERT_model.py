
from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam

set_gelu('tanh')

def textcnn(inputs, kernel_initalizer):
    # 一维卷积
    cnn1 = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                               kernel_initializer=kernel_initalizer)(inputs) # shape=[batch_size, maxlen-2, 256]
    cnn1 = keras.layers.GlobalMaxPool1D()(cnn1)     # shape=[batch_size, 256]

    cnn2 = keras.layers.Conv1D(filters=256, kernel_size=4, strides=1, padding='same',activation='relu',
                               kernel_initializer=kernel_initalizer)(inputs)
    cnn2 = keras.layers.GlobalMaxPool1D()(cnn2)  # shape=[batch_size, 256]

    cnn3 = keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='same',
                               kernel_initializer=kernel_initalizer)(inputs)
    cnn3 = keras.layers.GlobalMaxPool1D()(cnn3)  # shape=[batch_size, 256]

    output = keras.layers.concatenate(
        [cnn1, cnn2, cnn3],
        axis=-1)
    output = keras.layers.Dropout(0.2)(output)

    return output

def build_bert_model(config_path, checkpoint_path, class_nums):
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='bert',
        return_keras_model=False
    )
    # 如果后面不加Text-CNN的话就可以直接拿CLS去做分类的  后面直接+全连接层就可以了  shape=[batch_size, 768]
    cls_features = keras.layers.Lambda(lambda x: x[:, 0], name='cls_token')(bert.model.output)
    # 抽取除了第一个 和最后一个 之外 所有的token-embedding    shape=[batch_size, maxlen-2, 768]
    all_token_embedding = keras.layers.Lambda(lambda x: x[:, 1:-1], name='all_token')(bert.model.output)

    # shape=[batch_size, cnn_output_dim]
    cnn_features = textcnn(all_token_embedding, bert.initializer)
    # 将两个特征进行 拼接 起来
    concat_features = keras.layers.concatenate([cls_features, cnn_features], axis=-1)

    # 这边是加上 全连接层的   先设置 输出维度  就是标签的类型
    dense = keras.layers.Dense(
        units=512,
        activation='relu',
        kernel_initializer=bert.initializer
    )(concat_features)



    output = keras.layers.Dense(
        units=class_nums,
        # 设置激活函数
        activation='softmax',
        # api 里面的初始化
        kernel_initializer=bert.initializer
    )(concat_features)

    model = keras.models.Model(bert.model.input, output)
    print(model.summary())

    return model

if __name__ == '__main__':
    config_path = 'D:/Code/2022/python100day/KBQA-medical/BERT-TextCNN/bert_weight_files/bert_config.json'
    checkpoint_path = 'D://Code/2022/python100day/KBQA-medical/BERT-TextCNN/bert_weight_files/bert_model.ckpt'
    class_nums = 13
    build_bert_model(config_path, checkpoint_path, class_nums)