# Code-Switching Machine Translation from Kazakh to Russian 

Recent progress in NLP has spurred the development of technologies capable of handling code-switched data. Despite the initiation of Code-Switching research several years ago, progress within the research community has been sluggish. The primary challenges in addressing this issue arise from the insufficient availability of data. A limited number of languages, such as Spanish-English, Hindi-English, or Chinese-English, dominate research and resources in CSW. Nevertheless, numerous countries and cultures that extensively use CSW remain underrepresented in NLP research. 

The purpose of this project is to extend code-switching translation capabilities of models on low resource Kazakh - Russian language pair.

Here is  Code-Switching examples:
| Sentence                                                                                             |
|------------------------------------------------------------------------------------------------------|
| Кадымгы заварка (одноразовый емес) жылы суга шыгынын шыгарып аласыздар и сеуып койсаныз, иыс кетеди  |
| Ой тегі, любой адам солай істейді ғой, не болды сонша?!                                              |
| Айына 10-15кг ға дейін арықтау. Фигурная болғың келсе маған кел, Мен көмектесемін                    |

Wandb project: https://wandb.ai/maksim-borisov-2013/kk-ru-csw

# CSW modelling 
We employ different types of data augmentation to create code-switching (CSW) training data. Namely, 
- cs-1: Replace Kazakh word with Russian one in normal form.
- cs-2: Replace Kazakh word with Russian one's stem with ``Kazakh ending.''
- cs-3: Replace Kazakh word with Russian one in random form.
- cs-4: Replace Kazakh word with Russian word aligned using fastalign 
- cs-5: Replace Kazakh word with Russian word aligned using SimAlign

![image](https://github.com/BorisovMaksim/KazakhCSW/assets/44704968/1c1e0ce9-afad-457d-b490-21706a20e481)



# Contributions

- Low resource: 
  - Adding translated RTC corpus 
- Code switching
  - A new method of data augmentation for csw
  - A method that works on real data



## Results

Fine-tuning of existing models

![image](https://github.com/BorisovMaksim/KazakhCSW/assets/44704968/aea6626e-190e-4b1c-b56a-d5198dc9a607)

CSW modelling

![image](https://github.com/BorisovMaksim/KazakhCSW/assets/44704968/423f549c-08aa-48a3-8ad7-f78880bf1d7e)



