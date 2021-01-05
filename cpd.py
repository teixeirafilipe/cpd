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
# * Output a GnuPlot script for the plot
# * Save MatPlotLib's plot of PD (can be done from the interactive window)

"""
CPD - Complex Partial Dependence for Scikit-learn

CPD is a small Python module which generates partial dependence graphics and
tables for models.
"""

import csv
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import matplotlib.tri as tri

cite = "Filipe Teixeira. (2021, January 5). teixeirafilipe/cpd: Inital release (Version v0.1.0). Zenodo. http://doi.org/10.5281/zenodo.4419860"

bibtex = """
@software{filipe_teixeira_2021_4419860,
  author       = {Filipe Teixeira},
  title        = {teixeirafilipe/cpd: Inital release},
  month        = jan,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.4419860},
  url          = {https://doi.org/10.5281/zenodo.4419860}
}
"""

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
            # 2 dimensional PD two categorical variables
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
        elif ncf == 0 and nrf == 2:
            # 2 dimensional PD two real variables
            self._mode = '2DRPD'
            self._run_2DRPD(model, data, real_features)
        elif ncf == 0 and nrf > 1:
            # multi-dimensional real PD without search
            self._mode = 'MDRPD' 
            self._run_MDRPD(model, data, real_features)
        else:
            raise NotImplementedError("Requested combination of variables not implemented")
    def __repr__(self):
        return self._ascii()
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
    def _run_2DRPD(self, model, data, real_features):
        #x_names = self._search_features(data, cat_feature)
        #x_vals  = self._feature_cleanup(cat_feature, x_names) 
        response = list()
        # set the data so that all features share the same range
        #pseudo_data = data.copy()
        #pseudo_data.loc[0, real_feature]=data[real_feature].min()
        #pseudo_data.loc[1, real_feature]=data[real_feature].max()
        pdep = partial_dependence(model, data, real_features)
        for i,x in enumerate(pdep[1][0]):
            for j,y in enumerate(pdep[1][1]):
                response.append([x,y,pdep[0][0][i,j]])
        # expose data to the object's namespace
        self.response = response
        self.x_name = real_features[0].capitalize()
        self.x_vals = pdep[1][0]
        self.y_name = real_features[1].capitalize()
        self.y_vals = pdep[1][1]
    def _run_MDRPDWS(self, model, data, feature_key):
        feature_names = self._search_features(data,feature_key)
        self._run_MDRPD(model, data, self._search_features(data,feature_key))
        self.x_name = feature_key.capitalize()
        self.y_name = feature_key.capitalize() + ' Value'
    def _run_MDRPD(self, model, data, real_features):
        # set the data so that all features share the same range
        #pseudo_data = data.copy()
        #y_min =  np.inf
        #y_max = -np.inf
        #print(data)
        #for name in real_features:
        #    print(f"{name}: {min(data[name])} - {max(data[name])}")
        #    if min(data[name])< y_min: y_min = min(data[name])
        #    if max(data[name])> y_max: y_max = max(data[name])
        #print(y_min, y_max)
        #npoints = len(data)
        #for name in real_features:
        #    pseudo_data[name]=np.linspace(y_min,y_max,npoints)
        #    #pseudo_data[name][0]=y_min
        #    #pseudo_data[name][1]=y_max
        # Saving the response to a list x,y,model_response
        response = list()
        try:
            x_vals = [int(x.split('_')[-1]) for x in real_features]
        except:
            raise TypeError("Unsuported format of real variables.")
        for i, xn in enumerate(real_features):
            #pdep = partial_dependence(model, pseudo_data, [xn])
            pdep = partial_dependence(model, data, [xn])
            for j,ypos in enumerate(pdep[1][0]):
                response.append([x_vals[i], ypos, pdep[0][0][j]])
            #print(x_vals[i],min(r[1] for r in response),max(r[1] for r in response))
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
        elif self._mode=='MDRPDWS' or self._mode=='MDRPD' or self._mode=='2DRPD':
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
        elif self._mode=='MDRPDWS' or self._mode=='MDRPD' or self._mode=='2DRPD':
            for cw in col_widths:
                s += '-'*cw + ' '
            s += '\n'
            s += f"{self.x_name:^{col_widths[0]}s} "
            s += f"{self.y_name:^{col_widths[1]}s} "
            t="Model Response"
            s += f"{t:^{col_widths[2]}s} "
            s += '\n'
            for cw in col_widths:
                s += '-'*cw + ' '
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
    def plot(self, fn=None, **kwargs):
        fig, ax = plt.subplots()
        if self._mode=='1DCPD':
            bar = ax.bar(self.x_vals, self.response)
        elif self._mode=='2DCPD':
            cmap = kwargs['cmap'] if 'cmap' in kwargs else 'RdYlGn'
            im = ax.imshow(self.response, cmap=cmap)
            # For the heatmap, we need to transpose the x and y labels
            ax.set_xticks(np.arange(len(self.y_vals)))
            ax.set_yticks(np.arange(len(self.x_vals)))
            ax.set_xticklabels(self.y_vals)
            ax.set_yticklabels(self.x_vals)
            cbar = ax.figure.colorbar(im, ax=ax)
            ax.set_xlabel(self.y_name)
            ax.set_ylabel(self.x_name)
        elif self._mode=='2DCRPD':
            for cn in self.x_vals:
                xy = [r for r in self.response if r[0]==cn ]
                ax.plot([r[1] for r in xy],[r[2] for r in xy], label=cn)
            ax.set_xlabel(self.y_name)
            ax.set_ylabel("Model Response")
            ax.legend(title=self.x_name)
        elif self._mode=='MDRPDWS' or self._mode=='MDRPD':
            x_pts = [float(r[0]) for r in self.response]
            y_pts = [r[1] for r in self.response]
            z_pts = [r[2] for r in self.response]
            x_min = kwargs['xlim'][0] if 'xlim' in kwargs else min(x_pts)
            x_max = kwargs['xlim'][1] if 'xlim' in kwargs else max(x_pts)
            y_min = kwargs['ylim'][0] if 'ylim' in kwargs else min(y_pts)
            y_max = kwargs['ylim'][1] if 'ylim' in kwargs else max(y_pts)
            cmap = kwargs['cmap'] if 'cmap' in kwargs else 'RdYlGn'
            npoints = kwargs['npoints'] if 'npoints' in kwargs else 1024
            xi = np.linspace(x_min,x_max, npoints)
            yi = np.linspace(y_min,y_max, npoints)
            tt = tri.Triangulation(x_pts,y_pts)
            interpolator = tri.LinearTriInterpolator(tt, z_pts)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)
            grph = ax.contourf(xi, yi, zi, cmap= cmap)
            fig. colorbar(grph)
            ax.set_xlabel(self.x_name)
            ax.set_ylabel(self.y_name)
        elif self._mode=='2DRPD':
            x_pts = [r[0] for r in self.response]
            y_pts = [r[1] for r in self.response]
            z_pts = [r[2] for r in self.response]
            x_min = kwargs['xlim'][0] if 'xlim' in kwargs else min(x_pts)
            x_max = kwargs['xlim'][1] if 'xlim' in kwargs else max(x_pts)
            y_min = kwargs['ylim'][0] if 'ylim' in kwargs else min(y_pts)
            y_max = kwargs['ylim'][1] if 'ylim' in kwargs else max(y_pts)
            cmap = kwargs['cmap'] if 'cmap' in kwargs else 'RdYlGn'
            npoints = kwargs['npoints'] if 'npoints' in kwargs else 1024
            xi = np.linspace(x_min,x_max, npoints)
            yi = np.linspace(y_min,y_max, npoints)
            tt = tri.Triangulation(x_pts,y_pts)
            interpolator = tri.LinearTriInterpolator(tt, z_pts)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)
            grph = ax.contourf(xi, yi, zi, cmap= cmap)
            fig. colorbar(grph)
            ax.set_xlabel(self.x_name)
            ax.set_ylabel(self.y_name)
        else:
            raise NotImplementedError(f"Unknown mode: {self._mode}")
        if fn:
            plt.saveimage(fn)
        else:
            plt.show()
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
            elif self._mode=='MDRPDWS' or self._mode=='MDRPD' or self._mode=='2DRPD':
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
        


