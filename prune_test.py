from train_supervision import *
import torch_pruning as tp
from geoseg.models import UNetFormer

import loveda_test


def main():
    args = loveda_test.get_args()
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)
    model.cuda()
    model.eval()

    net = model.net
    net.eval()

    net: UNetFormer.UNetFormer
    image_inputs = torch.randn(2, 3, 1024, 1024).cuda()

    imp = tp.importance.RandomImportance()

    pruner = tp.pruner.MagnitudePruner(
        net,
        example_inputs=image_inputs,
        global_pruning=False,
        importance=imp,
        pruning_ratio=0.1,
        iterative_steps=1,
    )

    ori_macs, ori_size = tp.utils.count_ops_and_params(net, image_inputs)
    pruner.step()

    torch.save(net, "prune-10.pth")

    macs, size = tp.utils.count_ops_and_params(net, image_inputs)
    print("Origin", ori_macs, ori_size)
    print("New", macs, size)

    return

    # 2. Group coupled layers for model.conv1

    # 3. Prune grouped layers altogether
    if graph.check_pruning_group(group):  # avoid full pruning, i.e., channels=0.
        group.prune()

    print(model)


if __name__ == "__main__":
    main()
