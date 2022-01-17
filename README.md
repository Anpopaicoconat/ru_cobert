# ru_cobert
config

!wget https://huggingface.co/DeepPavlov/rubert-base-cased/resolve/main/config.json -P models/rubert/config

tokenizer

wget https://huggingface.co/DeepPavlov/rubert-base-cased/resolve/main/vocab.txt -P models/rubert/config
wget https://huggingface.co/DeepPavlov/rubert-base-cased/resolve/main/special_tokens_map.json -P models/rubert/config
wget https://huggingface.co/DeepPavlov/rubert-base-cased/resolve/main/tokenizer_config.json -P models/rubert/config

weights

wget https://huggingface.co/DeepPavlov/rubert-base-cased/resolve/main/pytorch_model.bin -P models/rubert/config

dataset

wget https://tlk.s3.yandex.net/dataset/TlkPersonaChatRus.zip
unzip TlkPersonaChatRus.zip -d /content/drive/MyDrive/stagirovka/
