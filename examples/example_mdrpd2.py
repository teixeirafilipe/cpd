#! /usr/bin/env python3

# MIT License
#
#Copyright 2020 Filipe Teixeira
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# from cpd import Partial_Dependence
# this is a workaround to avoid importing cpd
exec(open('../cpd.py','r').read())

data = pd.read_csv('compound_activity.csv')

y = data['Activity'].to_numpy()
X = pd.get_dummies(data.drop('Activity',axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

# This a very crude model withou any hyper-parameter optimization

rf_model = RandomForestRegressor()

rf_model.fit(X_train, y_train)

print(f"Score on the Train Set: {rf_model.score(X_train,y_train):6.4f}")
print(f"Score on the Test Set:  {rf_model.score(X_test,y_test):6.4f}")

print(X_train)

pd_data = Partial_Dependence(rf_model, X_train, real_features=[f"Spec_{x}" for x in range(200,221)])

print(pd_data.x_name)
print(pd_data.x_vals)

print(pd_data.y_name)
print(pd_data.y_vals)

print(pd_data.response)
