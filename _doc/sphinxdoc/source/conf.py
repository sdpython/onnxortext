# -*- coding: utf-8 -*-
import sys
import os
import alabaster
from pyquickhelper.helpgen.default_conf import set_sphinx_variables

sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0])))

set_sphinx_variables(__file__, "onnxortext", "Xavier Dupré", 2021,
                     "alabaster", alabaster.get_path(),
                     locals(), add_extensions=None,
                     extlinks=dict(issue=('https://github.com/sdpython/onnxortext/issues/%s', 'issue')))

blog_root = "http://www.xavierdupre.fr/app/onnxortext/helpsphinx/"
blog_background = False

html_css_files = ['my-styles.css']

nblinks = {}

epkg_dictionary = ({
    "onnx": 'https://github.com/onnx/onnx',
    "onnxruntime": 'https://github.com/microsoft/onnxruntime',
    "ort-customops": 'https://github.com/microsoft/ort-customops',
})
