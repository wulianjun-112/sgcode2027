import json

import os

with open('../docs/jiangsufangtian_sgcode.json','w') as f:
    aa = dict()

    aa['vendor'] = 'jiangsufangtian'
    aa['repositoryName '] = 'jiangsufangtian_sgcode'
    aa["gpuTypeList"] = ["RTX 2080Ti"]
    aa['model'] = []
    aa['model'].append({"code":'03',"name":'导地线','area':["华北", "华南", "华中", "东北"],"lineVoltage": ["1000kV","±500kV","±1100kV"]})