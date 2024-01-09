from datasets import Dataset, Audio
import pandas as pd
import json
import os
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, AutoFeatureExtractor
import IPython.display as ipd

if __name__ == '__main__':
  datasets_df = pd.read_csv('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data/final_dataset_df.csv', encoding='utf-8')
  audio_list = list(datasets_df['audio_path'])
  g2p_txt = list(datasets_df['g2p_text'])
  country = list(datasets_df['country'])

  ds = Dataset.from_dict({'audio' : audio_list,
                        "transcripts": g2p_txt,
                        "country" : country}).cast_column('audio', Audio(sampling_rate=16000))

  def extract_all_chars(batch):
    all_text = " ".join(batch["transcripts"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

  vocabs = ds.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=ds.column_names)
  vocab_list = list(set(vocabs["vocab"][0]))

  vocab_dict = {v: k for k, v in enumerate(vocab_list)}
  vocab_dict

  vocab_dict["|"] = vocab_dict[" "]
  del vocab_dict[" "]
  vocab_dict["[UNK]"] = len(vocab_dict)
  vocab_dict["[PAD]"] = len(vocab_dict)

  os.chdir('C:\\Users\\jjw28\\OneDrive\\바탕 화면\\wav2vec2\\data')
  with open('vocab.json', 'w') as vocab_file:
      json.dump(vocab_dict, vocab_file)

  tokenizer = Wav2Vec2CTCTokenizer("C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data/vocab.json", 
                                  unk_token="[UNK]", 
                                  pad_token="[PAD]", 
                                  word_delimiter_token="|")
  tokenizer.save_pretrained('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/model/custom_tokenizer')

  feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
                                               sampling_rate=16000, 
                                               padding_value=0.0, 
                                               do_normalize=True, 
                                               return_attention_mask=True)
  # feature_extractor = AutoFeatureExtractor.from_pretrained('kresnik/wav2vec2-large-xlsr-korean')

  processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
  processor.save_pretrained('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/model/custom_processor')

  
  print('-------------finish-------------') 