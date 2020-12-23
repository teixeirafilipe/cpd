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

#TODO:
# * Print PD as a LaTeX table
# * Plot PD using MatPlotLib
# * Save MatPlotLib's plot of PD
# * Output a GnuPlot script for the plot

"""
CPD - Complex Partial Dependence for Scikit-learn

CPD is a small Python module which generates partial dependence graphics and
tables for models.
"""

import csv
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
            self._mode = '2DCRPD'
            self._run_2DCRPD(model, data, cat_features[0], real_features[0])
        elif ncf == 0 and nrf == 1:
            # multi-dimensional real PD with search
            self._mode = 'MDRPDWS' 
            self._run_MDRPDWS(model, data, real_features[0])
        elif ncf == 0 and nrf > 1:
            # multi-dimensional real PD without search
            self._mode = 'MDRPD' 
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
        x_names = self._search_features(data, cat_feature)
        x_vals  = self._feature_cleanup(cat_feature, x_names) 
        response = list()
        # set the data so that all features share the same range
        pseudo_data = data.copy()
        pseudo_data.loc[0, real_feature]=data[real_feature].min()
        pseudo_data.loc[1, real_feature]=data[real_feature].max()
        for i, xn in enumerate(x_names):
            pdep = partial_dependence(model, pseudo_data, [xn,real_feature])
            for j,y_val in enumerate(pdep[1][1]):
                response.append([x_vals[i],y_val,pdep[0][0][1][j]])
        # expose data to the object's namespace
        self.response = response
        self.x_name = cat_feature.capitalize()
        self.x_vals = x_vals
        self.y_name = real_feature.capitalize()
        self.y_vals = None
    def _run_MDRPDWS(self, model, data, feature_key):
        feature_names = self._search_features(data,feature_key)
        self._run_MDRPD(model, data, self._search_features(data,feature_key))
        self.x_name = feature_key.capitalize()
        self.y_name = feature_key.capitalize() + ' Value'
    def _run_MDRPD(self, model, data, real_features):
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
        self.x_name = self._find_common_prefix(real_features)
        self.x_vals = x_vals
        self.y_name = self.x_name + ' Values'
        self.y_vals = None
    def _get_col_widths(self):
        o=[]
        if self._mode=='1DCPD':
            o.append(2+max(len(self.x_name),max([len(x) for x in self.x_vals])))
            o.append(2+len('Model Response'))
        elif self._mode=='2DCPD':
            o.append(2+max(len(self.x_name)+len(self.y_name),max([len(x) for x in self.x_vals])))
            for yv in self.y_vals:
                o.append(2+max(8,len(yv)))
        elif self._mode=='2DCRPD':
            o.append(2+max(8,len(self.y_name)))
            for xv in self.x_vals:
                o.append(2+max(8,len(xv)))
        elif self._mode=='MDRPDWS' or self._mode=='MDRPD':
            o.append(2+max(8,len(self.x_name)))
            o.append(2+max(8,len(self.y_name)))
            o.append(2+max(8,len("Model Response")))
        else:
            raise NotImplementedError(f"Unknown mode: {self._mode}")
        return o
    def _ascii(self, **kwargs):
        s=""
        col_widths = self._get_col_widths()
        if self._mode=='1DCPD':
            s += '-'*col_widths[0] + ' ' + '-'*col_widths[1] + '\n'
            s += f"{self.x_name:^{col_widths[0]}s}  Model Response\n"
            s += '-'*col_widths[0] + ' ' + '-'*col_widths[1] + '\n'
            for i,x in enumerate(self.x_vals):
                s += f"{x:^{col_widths[0]}s} {self.response[i]:+{col_widths[1]}.{col_widths[1]-5}g}\n"
            s += '-'*col_widths[0] + ' ' + '-'*col_widths[1] + '\n'
        elif self._mode=='2DCPD':
            for cw in col_widths:
                s += '-'*cw + ' '
            s += '\n'
            t = f"{self.x_name}/{self.y_name}"
            s += f"{t:^{col_widths[0]}s} "
            for i,cn in enumerate(self.y_vals):
                s += f"{cn:^{col_widths[i+1]}s} "
            s += '\n'
            for cw in col_widths:
                s += '-'*cw + ' '
            s += '\n'
            for i, rv in enumerate(self.x_vals):
                s += f"{rv:^{col_widths[0]}s} "
                for j, cv in enumerate(self.y_vals):
                    s += f"{self.response[i,j]:+{col_widths[j+1]}.{col_widths[j+1]-5}g} "
                s += '\n'
            for cw in col_widths:
                s += '-'*cw + ' '
            s += '\n'
        elif self._mode=='2DCRPD':
            for cw in col_widths:
                s += '-'*cw + ' '
            s += '\n'
            s += f"{self.y_name:^{col_widths[0]}s} "
            for i,xv in enumerate(self.x_vals):
                s += f"{xv:^{col_widths[i+1]}s} "
            s += '\n'
            for cw in col_widths:
                s += '-'*cw + ' '
            s += '\n'
            for i,yv in enumerate(list(set([x[1] for x in self.response]))):
                s += f"{yv:{col_widths[0]}.{col_widths[0]-5}g} "
                for j, xv in enumerate(self.x_vals):
                    r = [x for x in self.response if x[0]==xv and x[1]==yv]
                    if r:
                        s += f"{r[0][2]:+{col_widths[j+1]}.{col_widths[j-1]-5}g} "
                    else:
                        s += ' '*(col_widths[j+1]+1)
                s += '\n'
            for cw in col_widths:
                s += '-'*cw + ' '
            s += '\n'
        elif self._mode=='MDRPDWS' or self._mode=='MDRPD':
            for cw in col_widths:
                s += '-'*cw + ' '
            s += '\n'
            s += f"{self.x_name:^{col_widths[0]}s} "
            s += f"{self.y_name:^{col_widths[1]}s} "
            t="Model Response"
            s += f"{t:^{col_widths[2]}s} "
            s += '\n'
            for l in self.response:
                for i in range(len(col_widths)):
                    s += f"{l[i]:{col_widths[i]}.{col_widths[i]-5}g} "
                s += '\n'
            for cw in col_widths:
                s += '-'*cw + ' '
            s += '\n'
        else:
            raise NotImplementedError(f"Unknown mode: {self._mode}")
        return s
    def print_ascii(self, **kwargs):
        print(self._ascii(**kwargs))
    def __repr__(self):
        return self._ascii()
    def plot(self, fn, **kwargs):
        #TODO
        if self._mode=='1DCPD':
            pass
        elif self._mode=='2DCPD':
            pass
        elif self._mode=='2DCRPD':
            pass
        elif self._mode=='MDRPDWS' or self._mode=='MDRPD':
            pass
        else:
            raise NotImplementedError(f"Unknown mode: {self._mode}")
    def to_gnuplot(self, fn, **kwargs):
        #TODO
        if self._mode=='1DCPD':
            pass
        elif self._mode=='2DCPD':
            pass
        elif self._mode=='2DCRPD':
            pass
        elif self._mode=='MDRPDWS' or self._mode=='MDRPD':
            pass
        else:
            raise NotImplementedError(f"Unknown mode: {self._mode}")
    def to_latex(self, fn, **kwargs):
        #TODO
        if self._mode=='1DCPD':
            pass
        elif self._mode=='2DCPD':
            pass
        elif self._mode=='2DCRPD':
            pass
        elif self._mode=='MDRPDWS' or self._mode=='MDRPD':
            pass
        else:
            raise NotImplementedError(f"Unknown mode: {self._mode}")
    def to_csv(self, fn, **kwargs):
        with open(fn,'w', newline='') as f:
            csw = csv.writer(f)
            if self._mode=='1DCPD':
                csw.writerow([self.x_name,"Model Response"])
                for i,x in enumerate(self.x_vals):
                    csw.writerow([x,self.response[i]])
            elif self._mode=='2DCPD':
                csw.writerow([f"{self.x_name}/{self.y_name}"]+self.y_vals)
                for i,rv in enumerate(self.x_vals):
                    csw.writerow([rv]+[self.response[i,j] for j in range(len(self.y_vals))])
            elif self._mode=='2DCRPD':
                csw.writerow([f"{self.y_name}/{self.x_name}"]+self.x_vals)
                for i,yv in enumerate(list(set([x[1] for x in self.response]))):
                    row = [yv]
                    for j, xv in enumerate(self.x_vals):
                        r = [x for x in self.response if x[0]==xv and x[1]==yv]
                        if r:
                            row.append(r[0][2])
                        else:
                            row.append('NA')
                    csw.writerow(row)
            elif self._mode=='MDRPDWS' or self._mode=='MDRPD':
                csw.writerow([f"{self.x_name}", f"{self.y_name}", "Model Response"])
                for l in self.response:
                    csw.writerow(l)
            else:
                raise NotImplementedError(f"Unknown mode: {self._mode}")

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
        


