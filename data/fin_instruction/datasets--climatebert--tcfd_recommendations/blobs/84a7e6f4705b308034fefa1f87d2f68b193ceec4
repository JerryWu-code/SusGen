---
annotations_creators:
  - expert-generated
language_creators:
  - found
language:
  - en
license: cc-by-nc-sa-4.0
multilinguality:
  - monolingual
size_categories:
  - 1K<n<10K
source_datasets:
  - original
task_categories:
  - text-classification
task_ids: []
pretty_name: TCFDRecommendations
dataset_info:
  features:
  - name: text
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': none
          '1': metrics
          '2': strategy
          '3': risk
          '4': governance
  splits:
  - name: train
    num_bytes: 638487
    num_examples: 1300
  - name: test
    num_bytes: 222330
    num_examples: 400
  download_size: 492631
  dataset_size: 860817
---


# Dataset Card for tcfd_recommendations

## Dataset Description

- **Homepage:** [climatebert.ai](https://climatebert.ai)
- **Repository:**
- **Paper:** [papers.ssrn.com/sol3/papers.cfm?abstract_id=3998435](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3998435)
- **Leaderboard:**
- **Point of Contact:** [Nicolas Webersinke](mailto:nicolas.webersinke@fau.de)

### Dataset Summary

We introduce an expert-annotated dataset for classifying the TCFD recommendation categories ([fsb-tcfd.org](https://www.fsb-tcfd.org)) of paragraphs in corporate disclosures.

### Supported Tasks and Leaderboards

The dataset supports a multiclass classification task of paragraphs into the four TCFD recommendation categories (governance, strategy, risk management, metrics and targets) and the non-climate-related class.

### Languages

The text in the dataset is in English.

## Dataset Structure

### Data Instances

```
{
  'text': '− Scope 3: Optional scope that includes indirect emissions associated with the goods and services supply chain produced outside the organization. Included are emissions from the transport of products from our logistics centres to stores (downstream) performed by external logistics operators (air, land and sea transport) as well as the emissions associated with electricity consumption in franchise stores.',
  'label': 1
}
```

### Data Fields

- text: a paragraph extracted from corporate annual reports and sustainability reports
- label: the label (0 -> none (i.e., not climate-related), 1 -> metrics, 2 -> strategy, 3 -> risk, 4 -> governance)

### Data Splits

The dataset is split into:
- train: 1,300
- test: 400

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization

Our dataset contains paragraphs extracted from financial disclosures by firms. We collect text from corporate annual reports and sustainability reports.

For more information regarding our sample selection, please refer to the Appendix of our paper (see [citation](#citation-information)).

#### Who are the source language producers?

Mainly large listed companies.

### Annotations

#### Annotation process

For more information on our annotation process and annotation guidelines, please refer to the Appendix of our paper (see [citation](#citation-information)).

#### Who are the annotators?

The authors and students at Universität Zürich and Friedrich-Alexander-Universität Erlangen-Nürnberg with majors in finance and sustainable finance.

### Personal and Sensitive Information

Since our text sources contain public information, no personal and sensitive information should be included.

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

- Julia Anna Bingler
- Mathias Kraus
- Markus Leippold
- Nicolas Webersinke

### Licensing Information

This dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (cc-by-nc-sa-4.0). To view a copy of this license, visit [creativecommons.org/licenses/by-nc-sa/4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

If you are interested in commercial use of the dataset, please contact [markus.leippold@bf.uzh.ch](mailto:markus.leippold@bf.uzh.ch).

### Citation Information

```bibtex
@techreport{bingler2023cheaptalk,
    title={How Cheap Talk in Climate Disclosures Relates to Climate Initiatives, Corporate Emissions, and Reputation Risk},
    author={Bingler, Julia and Kraus, Mathias and Leippold, Markus and Webersinke, Nicolas},
    type={Working paper},
    institution={Available at SSRN 3998435},
    year={2023}
}
```

### Contributions

Thanks to [@webersni](https://github.com/webersni) for adding this dataset.