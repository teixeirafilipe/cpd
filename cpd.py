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

"""
CPD - Complex Partial Dependence for Scikit-learn

CPD is a small Python module which generates partial dependence graphics and
tables for models.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

class Partial_Dependence():
    def __init__(self, model, data, cat_features=[], real_features=[], **kwargs):
        ncf=len(cat_features)
        nrf=len(real_features)
        self._mode  = 'ND'
        if ncf == 1 and nrf == 0:
            # 1 dimensional PD 
            self._mode = '1DCPD'
            self._run_1DCPD(model,data,cat_features[0])
        elif ncf == 2 and nrf == 0:
            # 2 dimensional PD two real variables
            self._mode = '2DCPD'
            self._run_2DCPD(model,data,cat_features)
        elif ncf == 1 and nrf == 1:
            # 2 dimensional PD between a categorical and a real variable
            self.mode = '2DCRPD'
            self._run_2DCRPD(model, data, cat_features[0], real_features[0])
        elif ncf == 0 and nrf == 1:
            # multi-dimensional real PD with search
            self.mode = 'MDRPDWS' 
            self._run_MDRPDWS(model, data, real_features[0])
        elif ncf == 0 and nrf > 1:
            # multi-dimensional real PD without search
            self.mode = 'MDRPD' 
            self._run_MDRPD(model, data, real_features)
        else:
            raise NotImplementedError("Requested combination of variables not implemented")
    def _search_features(self, data, feature_key):
        return [x for x in data.columns if x.startswith(feature_key)]
    def _feature_cleanup(self, feature_key, feature_list):
        return [x.replace(f"{feature_key}_",'').capitalize() for x in feature_list]
    def _find_common_prefix(self, sl):
        o=''
        for i,c in enumerate(sl[0]):
            if all([s[i]==c for s in sl]):
                o += c
                continue
            else:
                break
        if o.endswith('_'): o = o[:-1]
        return o
    def _run_1DCPD(self, model, data, feature_key):
        x_names = self._search_features(data, feature_key)
        response = np.zeros(len(x_names))
        pdep = partial_dependence(model, data, x_names)
        for i, i_name in enumerate(x_names):
            s = 'pdep[0][0]'
            for ii in range(len(x_names)):
                if ii == i:
                    s += '[1]'
                else:
                    s += '[0]'
            response[i]=eval(s)
        # expose data to the object's namespace
        self.x_name = feature_key.capitalize()
        self.y_name = None
        self.x_vals = self._feature_cleanup(feature_key, x_names)
        self.y_vals = None
        self.response = response
    def _run_2DCPD(self, model, data, feature_keys):
        x_names = self._search_features(data, feature_keys[0])
        y_names = self._search_features(data, feature_keys[1])
        response = np.zeros((len(x_names),len(y_names)))
        pdep = partial_dependence(model, data, x_names + y_names)
        for i, i_name in enumerate(x_names):
            for j, j_name in enumerate(y_names):
                s = 'pdep[0][0]'
                for ii in range(len(x_names)):
                    if ii == i:
                        s += '[1]'
                    else:
                        s += '[0]'
                for jj in range(len(y_names)):
                    if jj == j:
                        s += '[1]'
                    else:
                        s += '[0]'
                response[i,j]=eval(s)
        # expose data to the object's namespace
        self.x_name = feature_keys[0].capitalize()
        self.y_name = feature_keys[1].capitalize()
        self.x_vals = self._feature_cleanup(feature_keys[0], x_names)
        self.y_vals = self._feature_cleanup(feature_keys[1], y_names)
        self.response = response
    def _run_2DCRPD(self, model, data, cat_feature, real_feature):
        raise NotImplementedError("This feature is not implemented.")
        pass
    def _run_MDRPDWS(self, model, data, feature_key):
        feature_names = self._search_features(data,feature_key)
        self._run_MDRPD(model, data, self._search_features(data,feature_key))
        self.x_name = feature_key.capitalize()
        self.y_name = feature_key.capitalize() + ' Value'
    def _run_MDRPD(self, model, data, real_features):
        print(real_features)
        # set the data so that all features share the same range
        pseudo_data = data.copy()
        bounds=( min(data.loc[:,tuple(real_features)].min()), max(data.loc[:,tuple(real_features)].max()))
        for name in real_features:
            pseudo_data[name][0]=bounds[0]
            pseudo_data[name][1]=bounds[1]
        # Saving the respons to a list x,y,model_response
        response = list()
        try:
            x_vals = [int(x.split('_')[-1]) for x in real_features]
        except:
            raise TypeError("Unsuported format of real variables.")
        for i, xn in enumerate(real_features):
            pdep = partial_dependence(model, pseudo_data, [xn])
            for j,ypos in enumerate(pdep[1][0]):
                response.append([x_vals[i], ypos, pdep[0][0][j]])
        # expose data to the object's namespace
        self.response = np.array(response)
        print('***')
        print(self._find_common_prefix(real_features))
        print('***')
        self.x_name = self._find_common_prefix(real_features)
        self.x_vals = x_vals
        self.y_name = self.x_name + ' Values'
        self.y_vals = None

def _main(**args):
    pass

if(__name__=='__main__'):
    # this is a stub, but may be used someday
    import sys
    opts={}
    if(len(sys.argv)<2):
        pass
        #print("Usage: {} [--notes] [--help]".format(sys.argv[0]))
        #sys.exit(0)
    n=1
    while(n<len(sys.argv)):
        if(sys.argv[n]=='--notes'):
            #print(notes)
            #sys.exit(0)
            pass
        elif(sys.argv[n]=='--help'):
            pass
            #print(__doc__)
            #sys.exit(0)
        else: # This might be the input file
            pass
            #print("Unknown argument: {}".format(sys.argv[n]))
            #sys.exit(1)
        n += 1
    _main(**opts)
        


