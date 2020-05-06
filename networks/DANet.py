from .danet.danet import DANet
def get_seg_model(args):
    model = DANet(nclass=args.num_classes, backbone='resnet101' ,dilated=True, 
                    multi_grid=True,multi_dilation=True)
    return model
