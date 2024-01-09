from datasets import Dataset, Audio
import IPython.display as ipd
import pandas as pd
from transformers import AutoTokenizer, AutoFeatureExtractor, Wav2Vec2Processor
# from jamo import h2j, j2hcj
from sklearn.model_selection import train_test_split
import jamo
import librosa

if __name__ =="__main__":
    feature_extractor = AutoFeatureExtractor.from_pretrained("42MARU/ko-spelling-wav2vec2-conformer-del-1s")
    feature_extractor.save_pretrained('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/model/feature_extractor_conformer')

    tokenizer = AutoTokenizer.from_pretrained("42MARU/ko-spelling-wav2vec2-conformer-del-1s")
    tokenizer.save_pretrained('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/model/feature_extractor_tokenizer')

    # beamsearch_decoder = build_ctcdecoder(
    #     labels=list(tokenizer.encoder.keys()),
    #     kenlm_model_path=None,
    # )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    
    datasets_df = pd.read_csv('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data/final_dataset_df.csv', encoding='utf-8')   
    _, datasets_df = train_test_split(datasets_df, stratify=datasets_df['country'], test_size = 0.1, random_state=23)
    datasets_df['text'] = datasets_df['text'].map(lambda x : ''.join(list(jamo.hangul_to_jamo(x))))
    datasets_df.to_csv('./used_dataset_df.csv', index=False, encoding='utf-8')

    audio_list = list(datasets_df['audio_path'])
    txt = list(datasets_df['text'])
    country = list(datasets_df['country'])

    ds = Dataset.from_dict({'audio' : audio_list,
                        "transcripts": txt,
                        "country" : country})
    ds = ds.class_encode_column("country")
    ds = ds.train_test_split(test_size=0.1, shuffle=True, stratify_by_column='country')

    def prepare_dataset(batch):
        audio = batch["audio"]
        raw, _ = librosa.load(batch['audio'], sr=16000)
        batch["input_values"] = processor(raw,sampling_rate=16000).input_values[0]
        # batch["input_length"] = len(batch["input_values"])
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcripts"]).input_ids
        return batch

    ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"])
    ds.save_to_disk('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/data/pre_datasets')

    print('토탈 시간 ', datasets_df['record_time'].sum())
    print('-------------finish-------------')