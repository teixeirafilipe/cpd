#! /usr/bin/env python3
# -*- coding: utf8 -*-
"cpd - A wrapper for more complex Partial Dependence plots in scikit-learn."
from distutils.core import setup

desc = dict()
desc['name']='cpd'
desc['version']='0.1.0'
desc['description']=__doc__,
desc['author']='Filipe Teixeira'
desc['author_email']="filipe.teixeira@fc.up.pt"
desc['url']="https://github.com/teixeirafilipe/cpd"
desc['maintainer']=desc['author']
desc['maintainer_email']=desc['author_email']
desc['keywords']=['Machine Learning','Partial Dependence','Scikit-learn']
desc['license'] = open('LICENSE','r').read()
desc['long_description'] = open('README.md','r').read()
desc['download_url']="https://github.com/teixeirafilipe/cpd"
desc['classifiers']=[
	"Development Status :: 3 - Alpha",
	"Environment :: Console",
	"Intended Audience :: Education",
	"Intended Audience :: End Users/Desktop",
	"Intended Audience :: Other Audience",
	"License :: OSI Approved :: MIT License",
	"Natural Language :: English",
	"Operating System :: POSIX :: Linux",
	"Programming Language :: Python :: 3",
	"Topic :: Scientific/Engineering",
	"Topic :: Scientific/Engineering :: Mathematics"
]
#desc['data_files']=[]
#desc['packages']=[]
desc['install_requires']=['numpy','scikit-learn','matplotlib','pandas']
desc['py_modules']=['cpd']
#desc['scripts']=[]
#desc['ext_modules']=[]
#desc['script_name']=''
#desc['script_args']=[]
#desc['options']=None
#desc['cmdclass']=None
#desc['package_dir']=''


setup(**desc)

