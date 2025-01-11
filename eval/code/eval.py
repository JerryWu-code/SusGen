# cd /home/whatx/SusGen/eval/code && CUDA_VISIBLE_DEVICES=1 python eval.py
import time
import os
import csv
from eval_finqa import evaluate_finqa
from eval_fintqa import evaluate_fintqa
from eval_hc import evaluate_headline_classification
from eval_ner import evaluate_ner
from eval_re import evaluate_re
from eval_sa import evaluate_sa
from eval_sum import evaluate_summarization
from eval_srg import evaluate_srg
from batch_inference import load_model
# def save_results_to_csv(results, df, output_dir, benchmark_name):
#     csv_path = os.path.join(output_dir, f"{benchmark_name}_eval_results.csv")
#     df.to_csv(csv_path, index=False)

#     txt_path = os.path.join(output_dir, f"{benchmark_name}_eval_results.txt")
#     with open(txt_path, 'w') as file:
#         # file.write(f"Benchmark: {benchmark_name}\n")
#         for metric, score in results.items():
#             file.write(f"{metric}: {score}\n")

def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.2f} seconds to execute.")
        return result
    return wrapper
          
def save_results_to_csv(results, df, output_dir, benchmark_name, test_data_path):
    csv_path = os.path.join(output_dir, f"{benchmark_name}_{os.path.basename(test_data_path)[:-5]}_eval_results.csv")
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(output_dir, f"{benchmark_name}_{os.path.basename(test_data_path)[:-5]}_eval_results.txt")
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
@time_logger
def main():
    # model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.3"
    # model_path = "../../ckpts/Meta-Llama-3-8B-Instruct-hf"
    
    prompt_type = 'mistral' # {'mistral', 'llama3}
    # model_name = model_path.split('/')[-1]
    model_name = 'Model2'
    summary_csv_path = f"../results/evaluation_summary_{model_name}.csv"
    lora_path = True # False
    lora_path = "/home/whatx/SusGen/results/SusGen30k-int4-adamw32_Mistral-7B-v0.3/checkpoint-1406"
    quantization = 'bf16' # {'int4', 'int8', 'bf16'}
    random_count = 5000 #5000
    batch_size = 128 #128
    # args = {
    #     # "max_length": 8096,
    #     "do_sample": True,
    #     "temperature": 0.2,
    #     "top_p": 0.9,
    #     "top_k": 40,
    #     "num_return_sequences": 1,
    #     "max_new_tokens": 256,
    # }
    args = {
        "temperature": 0.2,
        "max_tokens": 256,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.2,
        # 'max_seq_len': 512
    }

    # Test data paths
    test_data_paths = {
        'FINQA': ["../benchmark/FINQA/FinQA.json", "../benchmark/FINQA/FSRL.json"],
        'FINTQA': ["../benchmark/FINTQA/ConvFinQA.json", "../benchmark/FINTQA/TATQA.json"],
        'HC': ["../benchmark/HC/MultiFin.json","../benchmark/HC/MLESG.json"],
        'NER': ["../benchmark/NER/NER.json","../benchmark/NER/FINER-ORD.json"],
        'RE': ["../benchmark/RE/FinRED.json", "../benchmark/RE/SC.json"],
        'SA': ["../benchmark/SA/FiQA-SA.json", "../benchmark/SA/FOMC.json"],
        'SUM': ["../benchmark/SUM/EDTSum.json"],
        'SRG': ['../benchmark/SRG/Annual.json']
    }

    evaluation_functions = {
        'FINQA': evaluate_finqa,
        'FINTQA': evaluate_fintqa,
        'HC': evaluate_headline_classification,
        'NER': evaluate_ner,
        'RE': evaluate_re,
        'SA': evaluate_sa,
        'SUM': evaluate_summarization,
        'SRG': evaluate_srg
    }

    results_summary = {}
    model = load_model(model_path) # load_model(model_path, gpu_utilization=0.85)
    
    for benchmark_name, test_data_paths_list in test_data_paths.items():
        output_dir = f"../results/{model_name}/{benchmark_name}"
        os.makedirs(output_dir, exist_ok=True)

        # evaluate_function = evaluation_functions[benchmark_name]
        # results, df = evaluate_function(model_path, test_data_path, args, prompt_type, lora_path, quantization, random_count)

        # results_summary[benchmark_name] = results
        # save_results_to_csv(results, df, output_dir, benchmark_name)
        
        results_summary[benchmark_name] = {}
        # Check if results already exist
        results_exist = all(
            os.path.exists(os.path.join(output_dir, f"{benchmark_name}_{os.path.basename(test_data_path)[:-5]}_eval_results.csv"))
            for test_data_path in test_data_paths_list
        )
        if results_exist:
            print(f"Results for {benchmark_name} already exist. Skipping evaluation.")
            continue
        print('='*50)
        print('>>> Evaluating ', benchmark_name)
        for test_data_path in test_data_paths_list:
            evaluate_function = evaluation_functions[benchmark_name]

            time_logger_evaluation = time_logger(evaluate_function)
            results, df = time_logger_evaluation(model_path, test_data_path, args, prompt_type, lora_path, quantization, random_count, batch_size, model)

            results_summary[benchmark_name][os.path.basename(test_data_path)[:-5]] = results
            save_results_to_csv(results, df, output_dir, benchmark_name, test_data_path)
        # append summary to temp csv
        save_summary_to_csv(results_summary, f'temp_evaluation_summary_{model_name}.csv', model_name)
        
    # delete cache folder
    cache_folder = os.path.join('../results', model_path.split('/')[-1], 'cache')
    if os.path.exists(cache_folder):
        os.system(f"rm -r {cache_folder}")
    
    save_summary_to_csv(results_summary, summary_csv_path, model_name)
    print(f"Results saved to results/{model_name}/ and summary saved to {summary_csv_path}")

if __name__ == "__main__":
    main()
