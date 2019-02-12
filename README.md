Ludwig
======

Introduction
------------

让代码的归代码，让AI的归AI。

在不写代码就能进行AI开发的道路上，Uber今日又往前踏了一步。

刚刚，Uber宣布开源Ludwig，一个基于TensorFlow的工具箱。

有了它，不用写代码就能够训练和测试深度学习模型。

Uber表示，对于AI开发者来说，Ludwig可以帮助他们更好地理解深度学习方面的能力，并能够推进模型快速迭代。

对于AI专家来说，Ludwig可以简化原型设计和数据处理过程，从而让他们能够专注于开发深度学习模型架构。

训练只需数据文件和配置文件
Ludwig提供了一套AI架构，可以组合起来，为给定的用例创建端到端的模型。

开始模型训练，只需要一个表格数据文件（如CSV）和一个YAML配置文件——用于指定数据文件中哪些列是输入特征，哪些列是输出目标变量。

如果指定了多个输出变量，Ludwig将学会同时预测所有输出。

使用Ludwig训练模型，在模型定义中可以包含附加信息，比如数据集中每个特征的预处理数据和模型训练参数， 也能够保存下来，可以在日后加载，对新数据进行预测。

灵活组合，适用多种任务
对于Ludwig支持的数据类型（文本、图像、类别等），其提供了一个将原始数据映射到张量的编码器，以及将张量映射到原始数据的解码器。张量是线性代数中使用的数据结构。

内置的组合器，能够自动将所有输入编码器的张量组合在一起，对它们进行处理，并将其返回给输入解码器。

Uber表示，通过组合这些特定于数据类型的组件，用户可以将Ludwig用于各种任务。比如，组合文本编码器和类别解码器，就可以获得一个文本分类器。

每种数据类型有多个编码器和解码器。例如，文本可以用卷积神经网络（CNN），循环神经网络（RNN）或其他编码器编码。

用户可以直接在模型定义文件中指定要使用的参数和超参数，而无需编写单行代码。

基于这种灵活的编码器-解码器架构，即使是经验较少的深度学习开发者，也能够轻松地为不同的任务训练模型。

比如文本分类、目标分类、图像字幕、序列标签、回归、语言建模、机器翻译、时间序列预测和问答等等。

多种功能，不断拓展
为了让工具变得更好用，Ludwig还提供了各种工具：

用于训练、测试模型和获得预测的命令行程序；

用于评估模型并通过可视化比较预测结果的工具；

用于用户训练或加载模型，并获得对新数据预测的Python编程API。

此外，Ludwig还能够使用开源分布式培训框架Horovod，在多个GPU上训练模型，并快速迭代。

目前，Ludwig有用于二进制值，浮点数，类别，离散序列，集合，袋（bag），图像，文本和时间序列的编码器和解码器，并且支持选定的预训练模型。

Uber表示，未来将为每种数据类型添加几个新的编码器，比如用于文本的Transformer，ELMo和BERT，以及用于图像的DenseNet和FractalNet。

还将添加其他的数据类型，比如音频、点云和图形，同时集成更多可扩展的解决方案来管理大数据集，如Petastorm。
- Open Source: Apache License 2.0


Installation
------------

Ludwig's requirements are the following:

- numpy
- pandas
- scipy
- scikit-learn
- scikit-image
- spacy
- tensorflow
- matplotlib
- seaborn
- Cython
- h5py
- tqdm
- tabulate
- PyYAML

Ludwig has been developed and tested with Python 3 in mind.
If you don’t have Python 3 installed, install it by running:

```
sudo apt install python3  # on ubuntu
brew install python3      # on mac
```

At the time of writing this document, TensorFlow is not compatible with Python 3.7, so the recommended version of Python for Ludwig is 3.6.
You may want to use a virtual environment to maintain an isolated [Python environment](https://docs.python-guide.org/dev/virtualenvs/).

In order to install Ludwig just run:

```
pip install ludwig
python -m spacy download en
```

or install it by building the source code from the repository:

```
git clone git@github.com:uber/ludwig.git
cd ludwig
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en
python setup.py install
```

Beware that in the `requirements.txt` file the `tensorflow` package is the regular one, not the GPU enabled one.
To install the GPU enabled one replace it with `tensorflow-gpu`.

If you want to train Ludwig models in a distributed way, you need to also install the `horovod` and the `mpi4py` packages.
Please follow the instructions on [Horovod's repository](https://github.com/uber/horovod) to install it.


Basic Principles
----------------

Ludwig provides two main functionalities: training models and using them to predict.
It is based on datatype abstraction, so that the same data preprocessing and postprocessing will be performed on different datasets that share data types and the same encoding and decoding models developed for one task can be reused for different tasks.

Training a model in Ludwig is pretty straightforward: you provide a CSV dataset and a model definition YAML file.

The model definition contains a list of input features and output features, all you have to do is specify names of the columns in the CSV that are inputs to your model alongside with their datatypes, and names of columns in the CSV that will be outputs, the target variables which the model will learn to predict.
Ludwig will compose a deep learning model accordingly and train it for you.

Currently the available datatypes in Ludwig are:

- binary
- numeric
- category
- set
- bag
- sequence
- text
- timeseries
- image

The model definition can contain additional information, in particular how to preprocess each column in the CSV, which encoder and decoder to use for each one, feature hyperparameters and training parameters.
This allows ease of use for novices and flexibility for experts.


### Training

For example, given a text classification dataset like the following:

| doc_text                              | class    |
|---------------------------------------|----------|
| Former president Barack Obama ...     | politics |
| Juventus hired Cristiano Ronaldo ... | sport    |
| LeBron James joins the Lakers ...     | sport    |
| ...                                   | ...      |

you want to learn a model that uses the content of the `doc_text` column as input to predict the values in the `class` column.
You can use the following model definition:

```yaml
{input_features: [{name: doc_text, type: text}], output_features: [{name: class, type: category}]}
```

and start the training typing the following command in your console:

```
ludwig train --data_csv path/to/file.csv --model_definition "{input_features: [{name: doc_text, type: text}], output_features: [{name: class, type: category}]}"
```

and Ludwig will perform a random split of the data, preprocess it, build a WordCNN model (the default for text features) that decodes output classes through a softmax classifier, train the model on the training set until the accuracy on the validation set stops improving.
Training progress will be displayed in the console, but TensorBoard can also be used.

If you prefer to use an RNN encoder and increase the number of epochs you want the model to train for, all you have to do is to change the model definition to:

```yaml
{input_features: [{name: doc_text, type: text, encoder: rnn}], output_features: [{name: class, type: category}], training: {epochs: 50}}
```

Refer to the [User Guide](http://uber.github.io/ludwig/user_guide/) to find out all the options available to you in the model definition and take a look at the [Examples](http://uber.github.io/ludwig/examples/) to see how you can use Ludwig for several different tasks.

After training, Ludwig will create a directory under `results` containing the trained model with its hyperparameters and summary statistics of the training process.
You can visualize them using one of the several visualization options available in the `visualize` tool, for instance:

```
ludwig visualize --visualization learning_curves --training_stats results/training_stats.json
```

The commands will display a graph that looks like the following, where you can see loss and accuracy as functions of train iteration number:

![Learning Curves](https://raw.githubusercontent.com/uber/ludwig/master/docs/images/getting_started_learning_curves.png "Learning Curves")

Several visualizations are available, please refer to [Visualizations](http://uber.github.io/ludwig/user_guide/#visualizations) for more details.


### Distributed Training

You can distribute the training of your models using [Horovod](https://github.com/uber/horovod), which allows to train on a single machine with multiple GPUs as well as on multiple machines with multiple GPUs.
Refer to the [User Guide](http://uber.github.io/ludwig/user_guide/#distributed-training) for more details.


### Predict

If you have new data and you want your previously trained model to predict target output values, you can type the following command in your console:

```
ludwig predict --data_csv path/to/data.csv --model_path /path/to/model
```

Running this command will return model predictions and some test performance statistics if the dataset contains ground truth information to compare to.
Those can be visualized by the `visualize` tool, which can also be used to compare performances and predictions of different models, for instance:

```
ludwig visualize --visualization compare_performance --test_stats path/to/test_stats_model_1.json path/to/test_stats_model_2.json
```

will return a bar plot comparing the models on different measures:

![Performance Comparison](https://raw.githubusercontent.com/uber/ludwig/master/docs/images/compare_performance.png "Performance Comparison")

A handy `ludwig experiment` command that performs training and prediction one after the other is also available.


### Programmatic API

Ludwig also provides a simple programmatic API that allows you to train or load a model and use it to obtain predictions on new data:

```python
from ludwig import LudwigModel

# train a model
model_definition = {...}
model = LudwigModel(model_definition)
train_stats = model.train(training_dataframe)

# or load a model
model = LudwigModel.load(model_path)

# obtain predictions
predictions = model.predict(test_dataframe)

model.close()
```

`model_definition` is a dictionary contaning the same information of the YAML file.
More details are provided in the [User Guide](http://uber.github.io/ludwig/user_guide/) and in the [API documentation](http://uber.github.io/ludwig/api/).


Extensibility
=============

Ludwig is built from the ground up with extensibility in mind.
It is easy to add an additional datatype that is not currently supported by adding a datatype-specific implementation of abstract classes which contain functions to preprocess the data, encode it, and decode it.

Furthermore, new models, with their own specific hyperparameters, can be easily added by implementing a class that accepts tensors (of a specific rank, depending of the datatype) as inputs and provides tensors as output.
This encourages reuse and sharing new models with the community.
Refer to the [Developer Guide](http://uber.github.io/ludwig/developer_guide/) for further details.


Full documentation
------------------

You can find the full documentation [here](http://uber.github.io/ludwig/).
