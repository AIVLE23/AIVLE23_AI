from datasets import Dataset, Audio
import IPython.display as ipd
import pandas as pd
from transformers import Wav2Vec2Processor

if __name__ =="__main__":
    processor = Wav2Vec2Processor.from_pretrained('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/model/custom_processor')

    datasets_df = pd.read_csv('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data/final_dataset_df.csv', encoding='utf-8')
    audio_list = list(datasets_df['audio_path'])
    g2p_txt = list(datasets_df['g2p_text'])
    country = list(datasets_df['country'])

    ds = Dataset.from_dict({'audio' : audio_list,
                        "transcripts": g2p_txt,
                        "country" : country}).cast_column('audio', Audio(sampling_rate=16000))
    ds = ds.class_encode_column("country")
    ds = ds.train_test_split(test_size=0.1, shuffle=True, stratify_by_column='country')

    def prepare_dataset(batch):
        audio = batch["audio"]

        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcripts"]).input_ids
        return batch

    ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"])
    ds.save_to_disk('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data/pre_datasets')

    print('-------------finish-------------')