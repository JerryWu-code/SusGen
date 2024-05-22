# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to balance the dataset for using in the training llm

#############################################################################
import json, random, os

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

# shuffle and randomly sample from the json file
def sample_json(file, target_num=2000):
    data = load_json(file)
    data = random.sample(data, target_num)
    return data

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

target_dict = {
    "General": {
        "num": 3000,
        "data": [
            {
                "path": "/home/whatx/SusGen/data/susgen/alpaca/alpaca_data_gpt4.json",
                "target_num": 3000
            }
        ]
    },
    "Math": {
        "num": 3000,
        "data": [
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/grade-school-math/train.json",
                "target_num": 3000
            }
        ]
    },
    "HC": {
        "num": 3500,
        "data": [
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FinGPT/fingpt-headline/data/fingpt-headline_train.json",
                "target_num": 1500
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FinGPT/fingpt-headline-cls/data/fingpt-headline-cls_train.json",
                "target_num": 1500
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/PIXIU/flare-multifin-en.json",
                "target_num": 500
            }
        ]
    },
    "NER": {
        "num": 3500,
        "data": [
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FinGPT/fingpt-ner-cls/data/fingpt-ner-cls_train.json",
                "target_num": 2700
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FinGPT/fingpt-ner/data/fingpt-ner_train.json",
                "target_num": 500
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/PIXIU/flare-ner-train.json",
                "target_num": 300
            }
        ]
    },
    "RE": {
        "num": 3500,
        "data": [
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FinGPT/fingpt-finred-re/data/fingpt-finred-re_train.json",
                "target_num": 1750
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FinGPT/fingpt-finred-cls/data/fingpt-finred-cls_train.json",
                "target_num": 1750
            }
        ]
    },
    "SA": {
        "num": 3500,
        "data": [
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/esg_sentiment/esg_sentiment_train_final.json",
                "target_num": 843
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/enhanced_financial_phrasebank/sentiment_v2.json",
                "target_num": 1457
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FinGPT/fingpt-sentiment-train/data/fingpt-sentiment-train_train.json",
                "target_num": 800
            }
            ,
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FinGPT/fingpt-sentiment-cls/data/fingpt-sentiment-cls_train.json",
                "target_num": 400
            }
        ]
    },
    "FIN-QA": {
        "num": 3500,
        "data": [
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FINNUMBER/BQA-400.json",
                "target_num": 399
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FINNUMBER/EQA-400.json",
                "target_num": 400
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FINNUMBER/MCQA-400.json",
                "target_num": 398
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FINNUMBER/NQA-ARI-400.json",
                "target_num": 398
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FINNUMBER/NQA-COM-400.json",
                "target_num": 399
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FINNUMBER/NQA-EXT-400.json",
                "target_num": 397
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FinGPT/fingpt-fiqa_qa/data/fingpt-fiqa_qa_train.json",
                "target_num": 709
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/PIXIU/flare-finqa_train.json",
                "target_num": 400
            }
        ]
    },
    "FIN-TQA": {
        "num": 3500,
        "data": [
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/PIXIU/flare-convfinqa_train.json",
                "target_num": 2500
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/FinGPT/fingpt-convfinqa/data/fingpt-convfinqa_train.json",
                "target_num": 1000
            }
        ]
    },
    "SRG":{
        "num": 3000,
        "data": [
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/ESG_Chat/esg_data_final.json",
                "target_num": 914
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/tcfd_qa/tcfd_qa_v4_final.json",
                "target_num": 1669
            },
            {
                "path": "/home/whatx/SusGen/data/susgen/add_data/esg/esg_final.json",
                "target_num": 417
            }
        ]
    },
}

def sample():
    folder_d = "/home/whatx/SusGen/data/susgen/FINAL/PER_3500/"

    total = []
    for folder1, value in target_dict.items():
        folder = os.path.join(folder_d, folder1)
        for item in value['data']:
            target_num = item['target_num']
            file = item["path"]
            print(file)

            name = file.split("/")[-1].split(".json")[0]
            output = os.path.join(folder, f"{name}_{target_num}.json")
            data = sample_json(file, target_num=target_num)
            total.extend(data)
            save_json(data, output)
    save_json(
        total, 
        os.path.join(folder_d, f"FINAL_PER3500_{len(total) // 1000}k.json")
    )
    print(f"Total records: {len(total)}")

def main():
    sample()

if __name__ == "__main__":
    main()