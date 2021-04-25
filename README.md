# Cross-lingual Product Matching using Transformers

**Can matching knowledge be transferred between languages?**

- Can we augment performance on languages with small amounts of training data by adding data from more readily available languages (English)? (E.g. Fine-tune model for the task using english data **/** Further fine-tune using few examples of target language (few-shot learning))
- Can we achieve good performance without any data from the target language? (Zero-shot performance)
- If it works well, how can we explain it?
- 
### Contributors 
These people have contributed to this repository:
 - [@Progandi](https://github.com/Progandi)
 - [@bebing](https://github.com/bebing)
 - [@daniels9](https://github.com/daniels9)
 - [@fniesel](https://github.com/fniesel)
 - [@JakobGutmann](https://github.com/JakobGutmann)


## Setup
**How to use this Repository**

*Before installing, make sure to have [Microsoft Build Tools for C++](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/)  installed for fastText*
````bash
cd path/to/repo
conda env create -f environment.yml
conda activate team_project
````

**Data**
- Our datasets can be requested via mail at ralph@informatik.uni-mannheim.de, but are available for research purposes only.
