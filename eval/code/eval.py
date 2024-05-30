import os
import csv
from eval_finqa import evaluate_finqa
from eval_fintqa import evaluate_fintqa
from eval_hc import evaluate_headline_classification
from eval_ner import evaluate_ner
from eval_re import evaluate_re
from eval_sa import evaluate_sa
from eval_sum import evaluate_summarization

def save_results_to_csv(results, df, output_dir, benchmark_name):
    csv_path = os.path.join(output_dir, f"{benchmark_name}_eval_results.csv")
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(output_dir, f"{benchmark_name}_eval_results.txt")
    with open(txt_path, 'w') as file:
        # file.write(f"Benchmark: {benchmark_name}\n")
        for metric, score in results.items():
            file.write(f"{metric}: {score}\n")

def save_summary_to_csv(results, summary_csv_path, model_name):
    with open(summary_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Benchmark', 'Metric', 'Score'])
        for benchmark, metrics in results.items():
            for metric, score in metrics.items():
                writer.writerow([model_name, benchmark, metric, score])

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    prompt_type = 'mistral' # {'mistral', 'llama3}
    model_name = model_path.split('/')[-1]
    summary_csv_path = f"../results/evaluation_summary_{model_name}.csv"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }

    # Test data paths
    test_data_paths = {
        # 'FINQA': "../benchmark/FINQA/FinQA.json",
        'FINTQA': "../benchmark/FINTQA/ConvFinQA.json",
        # 'HC': "../benchmark/HC/MultiFin.json",
        # 'NER': "../benchmark/NER/NER.json",
        # # 'RE': "../benchmark/RE/fingpt-re_test.json",
        # 'SA': "../benchmark/SA/FiQA-SA.json",
        # 'SUM': "../benchmark/SUM/EDTSum.json"
    }

    evaluation_functions = {
        # 'FINQA': evaluate_finqa,
        'FINTQA': evaluate_fintqa,
        # 'HC': evaluate_headline_classification,
        # 'NER': evaluate_ner,
        # # 'RE': evaluate_re,
        # 'SA': evaluate_sa,
        # 'SUM': evaluate_summarization
    }

    results_summary = {}

    for benchmark_name, test_data_path in test_data_paths.items():
        output_dir = f"../results/{model_name}/{benchmark_name}"
        os.makedirs(output_dir, exist_ok=True)

        evaluate_function = evaluation_functions[benchmark_name]
        results, df = evaluate_function(model_path, test_data_path, args, prompt_type)

        results_summary[benchmark_name] = results
        save_results_to_csv(results, df, output_dir, benchmark_name)
        
        # append summary to temp csv
        save_summary_to_csv(results_summary, f'temp_evaluation_summary_{model_name}.csv', model_name)

    save_summary_to_csv(results_summary, summary_csv_path, model_name)
    print(f"Results saved to results/{model_name}/ and summary saved to {summary_csv_path}")

if __name__ == "__main__":
    main()
