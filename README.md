# SusGen

We curated a **well-balanced high-quality dataset in the financial & esg domain** and developed a suite of **data-centric Open-Source Large Language Models** for financial NLP tasks which are able to generate great sustainability report and perform multi financial tasks.

- **Target**: Training Large Language Models to advance **sustainablility report generation** especially for tcfd format report and also able to **perform multi financial NLP tasks**, like Headline Classification(**HC**), Named Entity Recognition(**NER**), Relation Extraction(**RE**), Sentiment Analysis(**SA**), Financial Question Answering(**FIN-QA**), Financial Tabel Question Answering(**FIN-TQA**), Text Summarisation(**SUM**), Sustainability Report Generation(**SRG**).

## Start with source-code

### Requirements

```
conda create -name susgen python==3.10
pip install -r requirements.txt
pip install deepspeed==0.31.1
pip install llama-index==0.10.31
pip install transformers accelerate bitsandbytes trl torch vllm tiktoken gradio
```

### Data Preparation

You could download the data in our huggingface repository via this [link](https://huggingface.co/datasets/WHATX/susgen-30k), and put it under the folder /data/susgen/.

### Training
Description to be finished.
```

```

### Evaluation
Description to be finished.
```

```

## About Team

<p align="center">
  <img src="assets/team_logo.png" width="240" height="240" alt="Team Logo">
</p>

| Team Leader                             | Team Member (Alphabet Order)                                 | Mentor                                                       |
| --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Wu Qilong](mailto:qilong_wu@u.nus.edu) | [Huang Hejia](mailto:e1124197@u.nus.edu)<br />[Wang Xuan](mailto:e1124070@u.nus.edu)<br />[Xiang Xiaoneng](mailto:e1124255@u.nus.edu) | [Dr. Satapathy Ranjan](mailto:satapathy_ranjan@ihpc.a-star.edu.sg)<br />[Prof. Bharadwaj Veeravalli](mailto:elebv@nus.edu.sg) |

## Acknowledgement

This project was sponsored by National University of Singapore and A*STAR Institute of High Performance.