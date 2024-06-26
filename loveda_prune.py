from train_supervision_distill import *
import torch_pruning as tp
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="")
    parser.add_argument("--name", default="swsl_resnet18")
    parser.add_argument("--ratio", type=float, help="pruning ratio", default=0.5)
    parser.add_argument("--importance", choices=["random", "mag", "bns", "lamp"], default="mag")
    parser.add_argument("--output", help="save file name", default="prune.ckpt")
    parser.add_argument("--verbose", action="store_true")
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
    config = py2cfg("GeoSeg/config/loveda/unetformer_distill.py")
    model = Supervision_Train.load_from_checkpoint(args.path, config=config)

    backbone = model.student_net.backbone
    backbone.eval()
    backbone.cpu()

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
        if args.verbose:
            print(group)

        group.prune()

    backbone.zero_grad()

    """
    print(backbone.feature_info.channels())

    new_sizes = [x.shape[1] for x in backbone(image_inputs)]

    # fix feature info
    feature_info = backbone.feature_info
    # print(feature_info.out_indices)

    for idx, size in zip(feature_info.out_indices, new_sizes):
        feature_info.info[idx]["num_chs"] = size

    print(backbone.feature_info.channels())
    torch.save(backbone, args.output)
    """

    new_sizes = [x.shape[1] for x in backbone(image_inputs)]
    backbone.feature_info = new_sizes
    torch.save(backbone, args.output)

    macs, size = tp.utils.count_ops_and_params(backbone, image_inputs)
    print("Origin", ori_macs, ori_size)
    print("New", macs, size)


if __name__ == "__main__":
    main()
