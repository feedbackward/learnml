# Learning to "do" machine learning

This project is composed of a series of short, hands-on tutorials designed to provide some insight into the multi-faceted nature of machine learning tasks. These days, it is very easy for almost anyone to fire up a standard library, pass some vector data to a pre-trained model, and have that pre-fab system carry out some predictions or quantization. In many cases, however, the existing models and algorithms may not suit the data/task at hand, at which point it becomes necessary to *make your own tools*. This requires a more intimate understanding of what is going on "behind the scenes" when standard libraries are being run.

This series of tutorials is designed to impart the key underlying ideas, design principles, and technical procedures involved in developing a "learning machine." We prioritize hands-on examples, with inline blocks of Python code available throughout, for users to read, modify, and execute themselves. We have also endeavoured to make lucid the correspondence between concepts illustrated using mathematical formulae, and the concrete objects that appear in the code.

__Author and maintainer:__<br>
<a href="http://feedbackward.com/">Matthew J. Holland</a> (Osaka University, Institute for Datability Science)


## Viewing the tutorial

Thanks to the superb nbviewer service, you can easily view static versions of the notebooks by following the links below. To actually modify the code and run it yourself, download the materials in whatever form you like, `cd` to the appropriate directory, and run `jupyter notebook` (in the case of running on a remote server, follow the instructions <a href="https://feedbackward.github.io/learnml/azure_use.html">here</a>).


#### Getting your hardware/software set up
- <a href="SetupYours.html">Working on your own machine</a>
- <a href="https://feedbackward.github.io/learnml/azure_use.html">Cloud services: Using Azure</a>

#### Getting data ready
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/DataSources.ipynb">Description of data sources</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/DataSourcesJPN.ipynb">JPN</a>)
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/DataMNIST.ipynb">Preparation of raw (binary) data</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/DataMNISTJPN.ipynb">JPN</a>)
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Datavim-2.ipynb">Preparation of raw data for encoding task</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Datavim-2JPN.ipynb">JPN</a>)

#### Preparing a model
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/FilterBank.ipynb">Building features using a Gabor filter bank</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/FilterBankJPN.ipynb">JPN</a>)

#### Designing a learning algorithm
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/AlgoIntro.ipynb">Basic format for learning algorithm design</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/AlgoIntroJPN.ipynb">JPN</a>)
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/AlgoSparseReg.ipynb">Algorithms for training sparse regression models</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/AlgoSparseRegJPN.ipynb">JPN</a>)

#### Doing some machine learning
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/FinishEncoder.ipynb">Finishing the encoder</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/FinishEncoderJPN.ipynb">JPN</a>)


All the main contents of this tutorial make use of data; some of it is simulated, and some of it is real. The real-world data must be acquired from the original sources. These are described and linked to in the <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/DataSources.ipynb">description of data sources</a> part of the tutorial, so please read this before starting.


## Downloading tutorial materials

If you have `git`, then the recommended procedure is to run:

```
git clone https://github.com/feedbackward/learnml.git
```

For those with some aversion towards git (for whatever reason), there are compressed archives available for direct download from this GitHub page as well.