# from .oc_module.asp_oc import get_resnet101_asp_oc_dsn
import importlib
import sys

def create_model(args):
    print('model is ', args.model)
    name = '.' + args.model
    mod = importlib.import_module(name,package='networks')
    model = mod.get_seg_model(args)   
    return model
