from train_supervision import *
import torch_pruning as tp
from geoseg.models import UNetFormer


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("--ratio", type=float, help="pruning ratio", default=0.5)
    arg("--cuda", action="store_true", help="run on gpu")
    arg("--importance", choices=["random", "mag", "bns", "lamp"], default="mag")
    arg("--output", help="save file name", default="prune.ckpt")
    return parser.parse_args()


def make_importance(importance):
    if importance == "random":
        return tp.importance.RandomImportance()
    elif importance == "mag":
        return tp.importance.MagnitudeImportance()
    elif importance == "bns":
        return tp.importance.BNScaleImportance()
    elif importance == "lamp":
        return tp.importance.LAMPImportance()


def main():
    args = get_args()
    importance = make_importance(args.importance)

    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)

    model.eval()

    net = model.net.backbone
    net.eval()

    ignored_layers = list()

    net: UNetFormer.UNetFormer
    image_inputs = torch.randn(2, 3, 1024, 1024)

    if args.cuda:
        net.cuda()
        image_inputs.cuda()

    pruner = tp.pruner.MagnitudePruner(
        net,
        example_inputs=image_inputs,
        global_pruning=False,
        importance=importance,
        pruning_ratio=args.ratio,
        iterative_steps=1,
        ignored_layers=ignored_layers,
    )

    ori_macs, ori_size = tp.utils.count_ops_and_params(net, image_inputs)
    for group in pruner.step(interactive=True):
        print(group)
        group.prune()

    torch.save(net, args.output)

    macs, size = tp.utils.count_ops_and_params(net, image_inputs)
    print("Origin", ori_macs, ori_size)
    print("New", macs, size)


if __name__ == "__main__":
    main()
