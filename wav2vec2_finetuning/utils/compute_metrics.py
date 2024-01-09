import nlptutti as metrics
import numpy as np

def compute_metrics(pred):
    wer_metric = 0
    cer_metric = 0
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    print(pred_str, label_str)
    
    for i in range(len(pred_str)):
        preds = pred_str[i].replace(" ", "")
        labels = label_str[i].replace(" ", "")
        wer = metrics.get_wer(pred_str[i], label_str[i])['wer']
        cer = metrics.get_cer(preds, labels)['cer']
        wer_metric += wer
        cer_metric += cer
        
    wer_metric = wer_metric/len(pred_str)
    cer_metric = cer_metric/len(pred_str)
    
    return {"wer": wer_metric, "cer": cer_metric}
