import os
import argparse
import numpy as np
import torch
import random
import logging
import sys

# 导入项目模块
from segment_anything_training import sam_model_registry
from model.mask_decoder_pa import MaskDecoderPA
from utils.dataloader import get_im_gt_name_dict, create_dataloaders, Resize
import utils.misc as misc

# 从 train.py 导入 evaluate 函数
# 确保 eval.py 和 train.py 在同一目录下
from train import evaluate


def get_default_args():
    parser = argparse.ArgumentParser('PA-SAM Evaluation Script', add_help=False)

    # ================= 配置区域 (在这里修改默认参数) =================

    # 1. 输出目录
    parser.add_argument("--output", type=str, default="work_dirs/pa_sam_l_eval",
                        help="输出结果和日志的目录")

    # 2. 模型类型 (vit_l, vit_b, vit_h)
    parser.add_argument("--model-type", type=str, default="vit_l")

    # 3. SAM 预训练权重路径
    parser.add_argument("--checkpoint", type=str, default="./pretrained_checkpoint/sam_vit_l_0b3195.pth",
                        help="SAM 原始权重")

    # 4. 我们训练好的模型路径 (要测试的权重)
    parser.add_argument("--restore-model", type=str, default="work_dirs/pa_sam_l/epoch_20.pth",
                        help="训练好的 PA-SAM 权重")

    # 5. 是否可视化 (生成掩码图片)
    parser.add_argument('--visualize', action='store_true', default=True,
                        help="是否保存预测结果图片")

    # ================= 系统默认参数 (通常不需要改) =================
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--find_unused_params', default=True)
    parser.add_argument('--eval', action='store_true', default=True)  # 强制为 Eval 模式
    parser.add_argument("--logfile", type=str, default=None)

    # 这里的 [] 表示不从命令行读取参数，而是使用上面的 default 值
    return parser.parse_args([])


def main():
    args = get_default_args()

    # ================= 你的修复代码：文件初始化 =================
    print("正在应用 Windows 分布式环境修复...")
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    # 确保使用 gloo (防止 utils/misc.py 没改对)
    # 注意：如果 utils/misc.py 里硬编码了 nccl，这里改 args 可能无效，
    # 但我们尽量尝试在这里覆盖，或者依赖你已经修改过的 misc.py
    args.dist_backend = 'gloo'

    init_file = os.path.abspath("dist_init_file_eval")  # 用个不一样的文件名
    if os.path.exists(init_file):
        try:
            os.remove(init_file)
        except OSError:
            pass

    args.dist_url = f"file:///{init_file.replace(os.sep, '/')}"
    print(f"Using init_method: {args.dist_url}")
    # ==========================================================

    # 初始化分布式环境
    misc.init_distributed_mode(args)
    print(f'World Size: {args.world_size}, Rank: {args.rank}')

    # 设置随机种子
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- 准备验证数据集 ---
    # 这里直接复制了 train.py 中的验证集配置，你可以根据需要注释掉没有的数据集

    # valid set
    dataset_coift_val = {"name": "COIFT",
                         "im_dir": "./data/thin_object_detection/COIFT/images",
                         "gt_dir": "./data/thin_object_detection/COIFT/masks",
                         "im_ext": ".jpg",
                         "gt_ext": ".png"}

    dataset_hrsod_val = {"name": "HRSOD",
                         "im_dir": "./data/thin_object_detection/HRSOD/images",
                         "gt_dir": "./data/thin_object_detection/HRSOD/masks_max255",
                         "im_ext": ".jpg",
                         "gt_ext": ".png"}

    dataset_thin_val = {"name": "ThinObject5k-TE",
                        "im_dir": "./data/thin_object_detection/ThinObject5K/images_test",
                        "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test",
                        "im_ext": ".jpg",
                        "gt_ext": ".png"}

    dataset_dis_val = {"name": "DIS5K-VD",
                       "im_dir": "./data/DIS5K/DIS-VD/im",
                       "gt_dir": "./data/DIS5K/DIS-VD/gt",
                       "im_ext": ".jpg",
                       "gt_ext": ".png"}

    # 修改列表：只保留你想跑的数据集
    valid_datasets = [dataset_dis_val]
    # valid_datasets = [dataset_dis_val, dataset_hrsod_val, dataset_thin_val]
    # 如果你有 COIFT 数据，可以把 dataset_coift_val 加回去

    print("--- 创建验证集 Dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                           my_transforms=[
                                                               Resize([1024, 1024])  # 默认尺寸
                                                           ],
                                                           batch_size=args.batch_size_valid,
                                                           training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    # --- 模型初始化 ---
    print(f"正在加载模型: {args.model_type}...")

    # 1. 定义 PA-Decoder
    net = MaskDecoderPA(args.model_type)
    if torch.cuda.is_available():
        net.cuda()

    # 包装 DDP (这是必须的，因为 evaluate 函数里可能用了 .module)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu],
                                                    find_unused_parameters=args.find_unused_params)
    net_without_ddp = net.module

    # 2. 定义 SAM
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu],
                                                    find_unused_parameters=args.find_unused_params)

    # 3. 加载我们训练好的权重 (Restore)
    if args.restore_model:
        print("正在恢复权重:", args.restore_model)
        if os.path.isfile(args.restore_model):
            if torch.cuda.is_available():
                checkpoint = torch.load(args.restore_model)
            else:
                checkpoint = torch.load(args.restore_model, map_location="cpu")

            # 这里的逻辑参考 train.py，根据 key 匹配加载
            net_without_ddp.load_state_dict(checkpoint, strict=False)
            print("权重加载成功！")
        else:
            print(f"警告: 找不到权重文件 {args.restore_model}")

    # --- 开始评估 ---
    print("开始执行 evaluate...")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 调用 train.py 中的 evaluate 函数
    evaluate(args, net, sam, valid_dataloaders, args.visualize, print_func=print)

    print(f"评估完成！结果已保存至 {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 暂停窗口，防止双击运行后直接闪退
        input("\n按回车键退出程序...")