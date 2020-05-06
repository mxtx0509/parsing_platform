from .cc_attention.ccnet import CCNet_Deeplab
def get_seg_model(args):
    model = CCNet_Deeplab(args.num_classes)
    return model
