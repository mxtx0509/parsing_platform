import argparse
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    BATCH_SIZE = 8
    DATA_DIRECTORY = '/RAID_20T/RAID_2/zm/LIP/'
    IGNORE_LABEL = 255
    INPUT_SIZE = '480,480'
    LEARNING_RATE = 1e-2
    MOMENTUM = 0.9
    NUM_CLASSES = 20
    POWER = 0.9
    RANDOM_SEED = 1234
    SNAPSHOT_DIR = './checkpoints/'
    WEIGHT_DECAY = 5e-4
    parser = argparse.ArgumentParser(description="CE2P Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--dataset", type=str, default='train', choices=['train', 'val', 'trainval', 'test'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default='', metavar='PATH',
                        help="Where restore model parameters from.")
    parser.add_argument("--save_step", type=int, default=2,
                        help="How many images to save.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--list_path", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--model", type=str, default='seg_hrnet',
                        help="choose model")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--epochs", type=int, default=150,
                        help="choose the number of recurrence.")
    parser.add_argument("--iouloss", action="store_true",
                        help="iouloss.")
    parser.add_argument("--ohem", action="store_true",
                        help="ohem.")
    parser.add_argument("--dsn", action="store_true",
                        help="dsn.")
    return parser.parse_args()


