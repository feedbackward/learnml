# Learning to "do" machine learning

This project is composed of a series of short, hands-on tutorials designed to provide some insight into the multi-faceted nature of machine learning tasks. These days, it is very easy for almost anyone to fire up a standard library, pass some vector data to a pre-trained model, and have that pre-fab system carry out some predictions or quantization. In many cases, however, the existing models and algorithms may not suit the data/task at hand, at which point it becomes necessary to *make your own tools*. This requires a more intimate understanding of what is going on "behind the scenes" when standard libraries are being run.

This series of tutorials is designed to impart the key underlying ideas, design principles, and technical procedures involved in developing a "learning machine." We prioritize hands-on examples, with inline blocks of Python code available throughout, for users to read, modify, and execute themselves. We have also endeavoured to make lucid the correspondence between concepts illustrated using mathematical formulae, and the concrete objects that appear in the code.

__Author and maintainer:__<br>
<a href="http://feedbackward.com/">Matthew J. Holland</a> (Osaka University, Institute for Datability Science)


## Viewing the tutorial

Thanks to the superb nbviewer service, you can easily view static versions of the notebooks by following the links below. To actually modify the code and run it yourself, download the materials in whatever form you like, `cd` to the appropriate directory, and run `jupyter notebook` (in the case of running on a remote server, follow the instructions <a href="https://feedbackward.github.io/learnml/cloud_use.html">here</a>).

#### Getting your hardware/software set up
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/SetupYours.ipynb">Working on your own machine</a>
- <a href="https://feedbackward.github.io/learnml/cloud_use.html">Using cloud-based solutions</a>

#### Getting data ready
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/DataSources.ipynb">Description of data sources</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/DataSourcesJPN.ipynb">JPN</a>)
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Data_MNIST.ipynb">MNIST handwritten digits</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Data_MNIST_JPN.ipynb">JPN</a>)
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Data_CIFAR10.ipynb">CIFAR-10 tiny images</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Data_CIFAR10_JPN.ipynb">JPN</a>)
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Data_Misc.ipynb">Miscellaneous benchmark data sets</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Data_Misc_JPN.ipynb">JPN</a>)
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Data_vim-2.ipynb">vim-2: visual stimulus and BOLD response</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Data_vim-2_JPN.ipynb">JPN</a>)

#### Fundamentals of implementing learning algorithms
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/FrameworkIntro.ipynb">Framework for prototyping</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/FrameworkIntro_JPN.ipynb">JPN</a>)
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Classifiers.ipynb">Classifier models</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Classifiers_JPN.ipynb">JPN</a>)
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Algo_FirstOrder.ipynb">Practice with first-order algorithms</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Algo_FirstOrder_JPN.ipynb">JPN</a>)
- <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Algo_SparseReg.ipynb">Learning algorithms for sparse regression</a> (old <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Algo_SparseReg_JPN.ipynb">JPN</a>)

#### Applied topics
- Encoder learning
  - <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/FilterBank.ipynb">Building features using a Gabor filter bank</a> (<a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/FilterBankJPN.ipynb">JPN</a>)
  - <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/FinishEncoder.ipynb">Finishing the encoder</a> (old <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/FinishEncoderJPN.ipynb">JPN</a>)
- Re-constructing the experiments of Johnson and Zhang
  - <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/ChainerWorkshop.ipynb">Using deep learning API to expedite implementation</a>
  - <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/Learn_JZ.ipynb">SVRG and numerical tests</a>

All the main contents of this tutorial make use of data; some of it is simulated, and some of it is real. The real-world data must be acquired from the original sources. These are described and linked to in the <a href="http://nbviewer.jupyter.org/github/feedbackward/learnml/blob/master/DataSources.ipynb">description of data sources</a> part of the tutorial, so please read this before starting.


## Downloading tutorial materials

If you have `git`, then the recommended procedure is to run:

```
git clone https://github.com/feedbackward/learnml.git
```

For those with some aversion towards git (for whatever reason), there are compressed archives available for direct download from this GitHub page as well.