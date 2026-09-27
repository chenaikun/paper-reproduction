#!/bin/bash
# ============================================================
# U-Net 逐步逼近消融实验
# 从 Baseline 出发，每次只改一个变量，逐步逼近论文原方案
#
# 用法:
#   bash run_ablation.sh          # 运行全部 6 步
#   bash run_ablation.sh 2        # 只运行 Step 2
#   bash run_ablation.sh 0 3      # 运行 Step 0 ~ Step 3
# ============================================================

set -e

START=${1:-0}
END=${2:-5}

SCRIPT="train_ablation.py"

run_step() {
    local STEP=$1
    echo ""
    echo "============================================================"
    echo "  Step $STEP"
    echo "============================================================"

    case $STEP in
        0)
            # 起始点: Baseline (BN + Adam + CE+Dice + flip/rot)
            python $SCRIPT --name step0_baseline_adam_ce_dice \
                --model baseline --optimizer adam --loss ce_dice \
                --lr 5e-4 --epochs 60 --batch-size 1 --seed 42
            ;;
        1)
            # 去 Dice loss
            python $SCRIPT --name step1_baseline_adam_ce \
                --model baseline --optimizer adam --loss ce \
                --lr 5e-4 --epochs 60 --batch-size 1 --seed 42
            ;;
        2)
            # 换 SGD 优化器 (加梯度裁剪安全网)
            python $SCRIPT --name step2_baseline_sgd_ce \
                --model baseline --optimizer sgd --loss ce \
                --lr 0.01 --momentum 0.99 --grad-clip 1.0 \
                --epochs 60 --batch-size 1 --seed 42
            ;;
        3)
            # 加边界加权损失 (含类别平衡 w_c)
            python $SCRIPT --name step3_baseline_sgd_weighted_ce \
                --model baseline --optimizer sgd --loss weighted_ce \
                --lr 0.01 --momentum 0.99 --grad-clip 1.0 \
                --w0 10.0 --sigma 5.0 \
                --epochs 60 --batch-size 1 --seed 42
            ;;
        4)
            # 去 BatchNorm — 纯论文架构
            python $SCRIPT --name step4_original_sgd_weighted_ce \
                --model original --optimizer sgd --loss weighted_ce \
                --lr 0.01 --momentum 0.99 --grad-clip 1.0 \
                --w0 10.0 --sigma 5.0 \
                --epochs 60 --batch-size 1 --seed 42
            ;;
        5)
            # 弹性变形 — 论文完整增强策略
            python $SCRIPT --name step5_original_sgd_weighted_elastic \
                --model original --optimizer sgd --loss weighted_ce \
                --lr 0.01 --momentum 0.99 --grad-clip 1.0 \
                --w0 10.0 --sigma 5.0 --elastic-deform \
                --epochs 60 --batch-size 1 --seed 42
            ;;
        6)
            # 补救: 无BN + 降低学习率 10 倍 — 测试原架构在温和优化器下能否稳定
            python $SCRIPT --name step6_original_sgd_weighted_lr1e3 \
                --model original --optimizer sgd --loss weighted_ce \
                --lr 0.001 --momentum 0.99 --grad-clip 1.0 \
                --w0 10.0 --sigma 5.0 \
                --epochs 60 --batch-size 1 --seed 42
            ;;
        7)
            # 补救: 无BN + 低学习率 + 弹性变形 — 论文完整方案但用稳定的学习率
            python $SCRIPT --name step7_original_sgd_weighted_elastic_lr1e3 \
                --model original --optimizer sgd --loss weighted_ce \
                --lr 0.001 --momentum 0.99 --grad-clip 1.0 \
                --w0 10.0 --sigma 5.0 --elastic-deform \
                --epochs 60 --batch-size 1 --seed 42
            ;;
        8)
            # 补救: 无BN + 常规动量 0.9 — 隔离动量 0.99 这个变量
            python $SCRIPT --name step8_original_sgd09_weighted \
                --model original --optimizer sgd --loss weighted_ce \
                --lr 0.01 --momentum 0.9 --grad-clip 1.0 \
                --w0 10.0 --sigma 5.0 \
                --epochs 60 --batch-size 1 --seed 42
            ;;
        9)
            # 补救: 无BN + Adam + CE — 与 step1 唯一区别是去 BN,隔离 BN 的作用
            python $SCRIPT --name step9_original_adam_ce \
                --model original --optimizer adam --loss ce \
                --lr 5e-4 \
                --epochs 60 --batch-size 1 --seed 42
            ;;
        10)
            # 补救: 无BN + Adam + 极低学习率 — 排除学习率因素的最后对照
            python $SCRIPT --name step10_original_adam_lr1e4 \
                --model original --optimizer adam --loss ce \
                --lr 1e-4 \
                --epochs 60 --batch-size 1 --seed 42
            ;;
        *)
            echo "未知步骤: $STEP"
            exit 1
            ;;
    esac
}

for STEP in $(seq $START $END); do
    run_step $STEP
done

echo ""
echo "============================================================"
echo "  全部消融步骤完成！"
echo "============================================================"
