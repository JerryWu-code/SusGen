import json
import pandas as pd

corpus = {}
with open('./fiqa/source/corpus.jsonl', 'r') as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc['_id']] = doc['text']

queries = {}
with open('./fiqa/source/queries.jsonl', 'r') as f:
    for line in f:
        query = json.loads(line)
        queries[query['_id']] = query['text']

def generate_json(tsv_file, output_file):
    qrels = pd.read_csv(tsv_file, sep='\t')

    train_data = []
    grouped = qrels.groupby('query-id')
    for query_id, group in grouped:
        query_text = queries[str(query_id)]
        for _, row in group.iterrows():
            if row['score'] > 0:  
                doc_text = corpus[str(row['corpus-id'])]
                train_data.append({
                    "instruction": "Answer the following question based on the provided document.",
                    "input": f"Question: {query_text}\nDocument: {doc_text}",
                    "answer": "The answer is relevant based on the provided document."
                })

    with open(output_file, 'w') as f:
        json.dump(train_data, f, indent=4)

    print(f"Generated {len(train_data)} examples for {output_file}.")

generate_json('./fiqa/source/qrels/train.tsv', './fiqa/train_data.json')
generate_json('./fiqa/source/qrels/dev.tsv', './fiqa/dev_data.json')
generate_json('./fiqa/source/qrels/test.tsv', './fiqa/test_data.json')
