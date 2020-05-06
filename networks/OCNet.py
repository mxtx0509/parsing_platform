from .oc_module.asp_oc import get_resnet101_asp_oc_dsn

def get_seg_model(args):
    model = get_resnet101_asp_oc_dsn(args.num_classes)
    return model