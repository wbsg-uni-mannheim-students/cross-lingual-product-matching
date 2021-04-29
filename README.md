# Cross-lingual Product Matching using Transformers

In our work, we  seek to explore whether learned matching knowledge in the product matching domain can be transferred between languages. More specifically, we investigate whether high-resource languages, such as English, can be used to augment performance in scenarios where either no (zero-shot) or few (few-shot) examples are available in the target language. Towards that, we also study the effect of cross-lingual pre-training with models like mBERT or XLM-R and compare them to monolingual Transformer-architectures and simple baseline models.

### Contributors 
These people have contributed to this repository:
 - [@Progandi](https://github.com/Progandi)
 - [@bebing](https://github.com/bebing)
 - [@daniels9](https://github.com/daniels9)
 - [@fniesel](https://github.com/fniesel)
 - [@JakobGutmann](https://github.com/JakobGutmann)

### Data
Our datasets can be requested via mail at ralph@informatik.uni-mannheim.de, but are available for research purposes only.

## Setup
**How to use this Repository**

*Before installing, make sure to have [Microsoft Build Tools for C++](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/)  installed for py_entitymatching*

### Settings files

Individual experiments can be configured using the `.json` settings files. 
The `settings_template.json` provides an overview over the possible settings for the experiments.
Some settings are only avaible in the multi-class setup but not in the pair-wise case, and vice versa.

### Run Experiments
To run a experiment, make sure to provide the path of the individual `.json` settings file
as input agrument. For instance, to run the `settings_baseline_multi.json`, include the argument
*--i path_to_file\settings_baseline_multi.json*.
