from train_supervision import *
import torch_pruning as tp
from geoseg.models.UNetFormer import load_backbone


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="")
    parser.add_argument("--name", default="swsl_resnet18")
    parser.add_argument("--ratio", type=float, help="pruning ratio", default=0.5)
    parser.add_argument("--importance", choices=["random", "mag", "bns", "lamp"], default="mag")
    parser.add_argument("--output", help="save file name", default="prune.ckpt")
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

    backbone = load_backbone(args.path, args.name, True)
    if not args.path:
        torch.save(backbone, f"{args.name}-pretrained.ckpt")

    backbone.eval()

    ignored_layers = list()

    image_inputs = torch.randn(2, 3, 1024, 1024)

    pruner = tp.pruner.MagnitudePruner(
        backbone,
        example_inputs=image_inputs,
        global_pruning=False,
        importance=importance,
        pruning_ratio=args.ratio,
        iterative_steps=1,
        ignored_layers=ignored_layers,
    )

    ori_macs, ori_size = tp.utils.count_ops_and_params(backbone, image_inputs)
    for group in pruner.step(interactive=True):
        print(group)
        group.prune()

    backbone.zero_grad()
    torch.save(backbone, args.output)

    macs, size = tp.utils.count_ops_and_params(backbone, image_inputs)
    print("Origin", ori_macs, ori_size)
    print("New", macs, size)


if __name__ == "__main__":
    main()
