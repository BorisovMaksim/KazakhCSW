# Code-Switching Machine Translation from Kazakh to Russian 
Wandb project: https://wandb.ai/maksim-borisov-2013/kk-ru-csw

## Results

### Pretrained models  
| Model                                    | ru -> kk | cs(ru) -> kk | kk -> ru | cs(kk) -> ru |
|------------------------------------------|----------|--------------|----------|--------------|
| baseline                                 |          |              | 6.12     | 7.01         |
| identity                                 | 1.15     | 30.99        | 1.16     | 7.55         |
| facebook/m2m100_418M                     | 0.21     | 8.43         | 1.46     | 5.39         |
| facebook/mbart-large-50-many-to-many-mmt | 0.87     | 22.93        | 2.47     | 4.62         |
| facebook/nllb-200-distilled-600M         | 6.75     | 25.30        | 10.94    | 12.26        |
| facebook/nllb-200-3.3B                   | 8.91     | 31.41        | 13.50    | 15.23        |
| facebook/nllb-moe-54b                    | 10.94    |              | 14.04    |              |
| google translation api (unofficial)      | 8.12     | 2.00         | 10.52    | 4.66         |


### Fine-tuned models

| Model                     | cs(kk) -> ru | fine-tune data |
|---------------------------|--------------|----------------|
| baseline                  | 7.0          | nu             |
| baseline                  | 5.7          | nu + cs-1      |
| baseline                  | 6.6          | nu + cs-2      |
| baseline                  | 6.0          | nu + cs-3      |
| facebook/mbart-large-cc25 | 6.1          | None           |
| facebook/mbart-large-cc25 | 6.8          | nu             |
| facebook/mbart-large-cc25 | 6.0          | nu + cs-1      |
| facebook/mbart-large-cc25 |              | nu + cs-2      |
| facebook/mbart-large-cc25 |              | nu + cs-3      |
| facebook/m2m100_418M      | 4.9          | None           |
| facebook/m2m100_418M      | 10.8         | nu             |
| facebook/m2m100_418M      | 9.5          | nu + cs-1      |
| facebook/m2m100_418M      | 10.2         | nu + cs-2      |
| facebook/m2m100_418M      | 8.9          | nu + cs-3      |

### Experiments
| Model                                  | nu (test) | kaznu (test) | ntrex | mix_test | ted   |
|----------------------------------------|-----------|--------------|-------|----------|-------|
| exp_transformer                        | 37.29     | 33.11        | 8.05  | 2.87     | 6.60  |
| exp_transformer_all_data_no_RTC        | 39.14     | 38.67        | 11.47 | 7.63     | 12.35 |
| exp_transformer_all_data               | 36.46     | 36.75        | 13.32 | 9.55     | 12.10 |
| exp_transformer_all_data_bi            | 20.48     | 33.43        | 12.26 | 5.31     | 10.99 |
| exp_bigger_transformer_all_data_no_RTC | 24.50     | 16.11        | 5.29  | 4.40     | 3.80  |


