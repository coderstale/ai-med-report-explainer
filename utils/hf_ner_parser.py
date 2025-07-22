from transformers import pipeline
import torch

pipeline_task = "ner"
model_name = "d4data/biomedical-ner-all"

ner = pipeline(
    pipeline_task,
    model=model_name,
    tokenizer=model_name,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

def extract_lab_results_from_ner(text):
    try:
        results = ner(text)
        output = []
        for entity in results:
            if entity['entity_group'] in ['VALUE', 'TEST']:
                output.append({
                    'name': entity['word'],
                    'value': None,
                    'unit': ""
                })
        return output
    except Exception as e:
        print(f"NER extraction failed: {e}")
        return []
