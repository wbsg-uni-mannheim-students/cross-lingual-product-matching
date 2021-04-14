# Cross-lingual Product Matching using Transformers

# Project Goal
**Can matching knowledge be transferred between languages?**

- Can we augment performance on languages with small amounts of training data by adding data from more readily available languages (English)? (E.g. Fine-tune model for the task using english data **/** Further fine-tune using few examples of target language (few-shot learning))
- Can we achieve good performance without any data from the target language? (Zero-shot performance)
- If it works well, how can we explain it?

# Setup
**Use this Repository**

*Before installing, make sure to have [Microsoft Build Tools for C++](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/)  installed for fastText*
````bash
cd path/to/repo
conda env create -f environment.yml
conda activate team_project
````
**Colab**

- To use GitHub-Repos in Colab, when opening new notebooks, you have to -> check the box for "include private repositories" ("private Repositories einschließen" in German). Then, Colab has to be given the necessary access rights to one's GitHub-Account (via a popup).
- To push changes: in a notebook -> Datei -> "Kopie in GitHub speichern" -> write useful commit message.

**Data**
- Datasets can be downloaded from
