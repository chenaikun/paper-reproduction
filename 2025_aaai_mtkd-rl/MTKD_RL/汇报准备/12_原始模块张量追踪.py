"""用随机输入运行固定提交的原始模块；CPU 形状/梯度诊断，不复现准确率。

依赖 torch、numpy、einops。默认读取同目录 author_source/。
为适配 CPU，本脚本手动组合 train() 的主要运算；
原始模型、TransFeat、PolicyTrans、get_agent_state、loss 和 train_agent 未改动。
"""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import torch.nn.functional as F

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent / "author_source")
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from train_loops import get_agent_state, train_agent
    from utils import DistillKL
    from distiller_zoo import FeatureMSELoss
    regnet = load_module("trace_regnet", source / "models/regnet.py")
    resnet = load_module("trace_resnet", source / "models/resnet.py")
    wrn = load_module("trace_wrn", source / "models/wrn.py")
    shuffle = load_module("trace_shuffle", source / "models/ShuffleNetv2.py")
    util = load_module("trace_util", source / "models/util.py")
    policy = load_module("trace_policy", source / "models/policy.py")
    torch.manual_seed(20261120)
    torch.set_num_threads(2)
    student = shuffle.ShuffleV2(num_classes=100)
    teachers = [regnet.RegNetY_400MF(100), regnet.RegNetX_400MF(100),
                resnet.resnet32x4(num_classes=100), wrn.wrn_28_4(num_classes=100)]
    for teacher in teachers:
        teacher.eval()
        teacher.requires_grad_(False)
    student.train()
    x = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([2, 17])
    fs, zs = student(x, is_feat=True)
    with torch.no_grad():
        teacher_results = [t(x, is_feat=True) for t in teachers]
    tf = [r[0][-2] for r in teacher_results]
    te = [r[0][-1] for r in teacher_results]
    tz = [r[1] for r in teacher_results]
    aligner = util.TransFeat(fs[-2].shape, [f.shape for f in tf])
    trans = aligner(fs[-2])
    kd = DistillKL(4)
    mse_fn = FeatureMSELoss()
    state = get_agent_state(trans, te, zs, tz, targets, kd)
    widths = [v.shape[1] for v in state[0]]
    agent = policy.PolicyTrans(widths, 4, dynamic=False)
    agent.eval()
    with torch.no_grad():
        wl, wf = agent(state)
    ce = F.cross_entropy(zs, targets, reduction="none")
    kls = torch.stack([kd(zs, t, unreduce=True) for t in tz], dim=1)
    mses = torch.stack([mse_fn(s, t) for s, t in zip(trans, tf)], dim=1)
    loss = (ce + (wl*kls).sum(1) + 5*(wf*mses).sum(1)).mean()
    loss.backward()
    phase_student = {
        "student_has_grad": any(p.grad is not None for p in student.parameters()),
        "aligner_has_grad": any(p.grad is not None for p in aligner.parameters()),
        "agent_has_grad": any(p.grad is not None for p in agent.parameters()),
        "teacher_has_grad": any(p.grad is not None for t in teachers for p in t.parameters()),
    }
    optimizer = torch.optim.SGD(list(student.parameters())+list(aligner.parameters()), lr=.001)
    optimizer.step()
    # 源码使用更新前 forward 的缓存构造 reward。本段复制其数值路径，
    # 用 no_grad 避免保存已不需要的图；未给学生重新 forward。
    with torch.no_grad():
        reward = -(F.cross_entropy(zs, targets, reduction="none")
                   + sum(wl[:,i]*kd(zs, tz[i], unreduce=True) for i in range(4))
                   + 5*sum(wf[:,i]*mse_fn(trans[i], tf[i]) for i in range(4)))
        normalized = ((reward-reward.mean())/reward.std()).detach().clamp(0,1)
    optimizer.zero_grad(set_to_none=True)
    agent_optimizer = torch.optim.SGD(agent.parameters(), lr=.1)
    train_agent(SimpleNamespace(rank=1), 1, [state], [normalized], [wl], agent, agent_optimizer)
    phase_agent = {
        "student_has_grad": any(p.grad is not None for p in student.parameters()),
        "aligner_has_grad": any(p.grad is not None for p in aligner.parameters()),
        "agent_has_grad": any(p.grad is not None for p in agent.parameters()),
        "teacher_has_grad": any(p.grad is not None for t in teachers for p in t.parameters()),
    }
    result = {
        "commit": "4e2b62adb2bab0d3fde4118e19c4ac2fb8006991",
        "torch_version": torch.__version__,
        "boundary": "CPU B=2; random inputs and random model weights; selected original modules only; no accuracy reproduction",
        "inputs": list(x.shape), "targets": list(targets.shape),
        "student_feature_list": [list(f.shape) for f in fs],
        "student_logits": list(zs.shape),
        "teacher_feature_maps": [list(f.shape) for f in tf],
        "teacher_embeddings": [list(f.shape) for f in te],
        "teacher_logits": [list(f.shape) for f in tz],
        "trans_student_features": [list(f.shape) for f in trans],
        "teacher_info_widths": widths,
        "teacher_info_shapes": [list(f.shape) for f in state[0]],
        "state_other_shapes": [list(f.shape) for f in state[1:]],
        "concatenated_input_width": sum(widths), "hidden_width": agent.steam[0].out_features,
        "logit_actions": list(wl.shape), "feature_actions": list(wf.shape),
        "action_row_sums": wl.sum(1).tolist(),
        "ce_shape": list(ce.shape), "kd_matrix_shape": list(kls.shape),
        "mse_matrix_shape": list(mses.shape), "total_loss_shape": list(loss.shape),
        "reward_shape": list(reward.shape), "normalized_reward_shape": list(normalized.shape),
        "student_phase_gradients": phase_student, "agent_phase_gradients": phase_agent,
        "random_total_loss": loss.item(),
        "random_raw_reward": reward.tolist(), "random_normalized_reward": normalized.tolist(),
    }
    assert sum(widths) == 1692 and widths == [487,487,359,359]
    assert [list(f.shape[1:]) for f in trans] == [[384,8,8],[384,8,8],[256,8,8],[256,8,8]]
    assert torch.allclose(wl.sum(1), torch.ones(2))
    assert all(not item.requires_grad for item in [*state[0],*state[1:]])
    assert phase_student == {"student_has_grad":True,"aligner_has_grad":True,"agent_has_grad":False,"teacher_has_grad":False}
    assert phase_agent == {"student_has_grad":False,"aligner_has_grad":False,"agent_has_grad":True,"teacher_has_grad":False}
    assert torch.isfinite(normalized).all()
    (Path(__file__).resolve().parent/"原始模块追踪结果.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
