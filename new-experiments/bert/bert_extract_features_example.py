import keras
from keras_bert import get_custom_objects, extract_embeddings, get_base_dict

model_path = 'keras_bert.h5'

model = keras.models.load_model(model_path, custom_objects=get_custom_objects())

sentence_pairs = [
    [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
    [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
    [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
]

token_dict = get_base_dict()
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())

model.summary(line_length=200)

texts = ['all work and no play', 'makes jack a dull boy~']

embeddings = extract_embeddings(model, texts, vocabs=token_dict)
# print(embeddings)
