# CPD - Complex Partial Dependence for Scikit-learn


## About PhiMD

CPD is a small Python module which generates partial dependence graphics and
tables for models.

## Motivation and Use Cases
Many Machine Learning models manipulate categorical variables which are
typically encoded using one-hot encoding, thus transforming each of such
variables into a set of logical variables (one for each of the categories in the
original data). In order to have a proper glimpse on the model's partil response
to such a categorical variable, one must evaluate the partial dependence with
respect to each of the columns created by the one-hot encoding, while applying
adequate constraints on the other "sister columns".

CPD eases the construction of such partial dependence data, by using
[Scikit-learn](https://scikit-learn.org)'s implementation of the partial
dependence function (future versions might do a complete re-implementation of
this, for improved performance). CPD allows enquires of the partial dependence
in the following cases:
1. one categorical variable
2. two categorical variables
3. one categorical variable _versus_ one real variable
4. a set of real variables closely related

Examples of these four cases are given in the ```examples``` folder. The first
two use cases are self-evident. The third case compares the model's response
with respect to one variable for each of the possible values of a categorical
variable.

The final case has a prticular application in chemistry, but similar cases can
be conceived in other fields. Let's suppose our model predicts a given property
based on some spectroscopic data (for example, the infrared spectra). The
infrared data used in the model is usually a sampling of the absorvance at given
values of the wavelenght and might be represented by a large number of variables
named, for example, *IR_610*, *IR_611*, etc, were the number after the last
underscore marks the position in the spectrum. CPD allows one to have a "bird's
eye" view of the model's partial dependence on the IR sepctrum as a whole by
combining the individual responses of each of the *IR_* variables.

## System Requirements
PhiMD is written in Python, and should work with any recent python distribution,
although version 3.6 and above is highly recommended. It also needs some
additional modules, which can be intalled using pip or your operating system's
package management:
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit-learn](https://scikit-learn.org)

## Instalation
In order to use CPD, you just need to have a copy of cpd.py on your project's
folder and import it to your code. 

Alternatively, you can clone this repository and use setuptools:

```
python setup.py
```

If you whish to further edit this work, you might prefer to put a linked version
of ```cpd.py``` on your PYTHON_LIB_PATH, using:

```
cd cpd
pip install -e .
```

## How to Use
After importing, the ```cpd.Partial_Dependence``` class becomes available. This
class is initialized using a similar nomenclature to that of
[Scikit-learn](https://scikit-learn.org)'s


## How to Cite
TBA

