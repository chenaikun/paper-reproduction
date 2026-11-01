# 代码与框架



---

## 第 1 

实验是怎么搭建

```
┌─────────────────────────────────────────────┐
│  官方 LD 仓库（MMDetection 2.10，submodule）  │  ← 一行没改，只打补丁
├─────────────────────────────────────────────┤
│  ld_core_experiment（写的壳）           │  ← 只改配置、传参、启进程
├─────────────────────────────────────────────┤
│  手写练习（KL / VLR / dim）                   │  ← 不参与训练，只验证理解
└─────────────────────────────────────────────┘
```

## 第 2     MMDetection

MMDetection = 注册表 + 配置 + 构建器

```
   配置（Python dict）              注册表（花名册）           对象
   ─────────────────              ─────────────────         ─────────
   model = dict(                  @DETECTORS.register_module()
     type='GFL',        ──────►     class GFL(...)     ──►   nn.Module
     backbone=dict(                @HEADS.register_module()
       type='ResNet'                class GFLHead(...)
     )
   )
        │
        └── build_detector(cfg.model) 递归装配，每个 dict 变成一个 Module
```



> "MMDetection 的核心就三样东西。
>
> **第一，注册表。** 所有零件——骨干网、neck、head、loss——都通过一个装饰器登记在一个'花名册'里，每个零件有个字符串名字。
>
> **第二，配置。** 配置文件是一个纯 Python 文件，里面只有 dict，`type='GFL'` 就是'从花名册里挑 GFL 这个名字'。
>
> **第三，构建器。** `build_detector` 拿着这个 dict，去花名册里按名字查类，把剩下的键当参数传进去实例化；如果参数里还有 dict，就递归再来一遍。
>
> 所以**配置是图纸，build 是照着图纸装配**。改网络结构不用改代码，改 dict 就行——这是整个框架的设计前提。"


---

## 第 3  · config 的三个机制

`_base_` 继承、深度合并、config 就是 Python


```python
# configs/ld_r18_voc_kd_only.py  （全文 11 行）
_base_ = ['./ld_r18_voc.py']
model = dict(bbox_head=dict(
    loss_kd=dict(loss_weight=10, T=2),
    loss_ld=dict(loss_weight=0, T=10),        # ← 只改这个 0
    loss_ld_vlr=dict(loss_weight=0, T=10),    # ← 和这个 0
))
```

> 配置文件有三个机制
>
> **第一，`_base_` 是继承。** 这 11 行就定义了一个完整变体，因为上面那 100 多行模型结构全是从官方配置继承来的。
>
> **第二，合并是'深度合并'。** dict 是递归合并的，list 是整个替换。所以这里只写 `loss_weight=0`，其他键（`type`、`T`）都还在。
>
> **第三，config 是当 Python 代码执行的，不是当数据解析的。**

---

## 第 4  · 一条命令

```
python scripts/train_ld.py --variant loc-only --teacher-checkpoint xxx.pth
   │
   ├─ ① 检查教师权重存在？ → 不存在 3 秒内退出
   ├─ ② build_env()：把 profile 展开成 10 个环境变量
   │     （VOC_DATA_ROOT / LD_PROFILE / SAMPLES_PER_GPU / ...）
   ▼
subprocess: python LD/tools/train.py configs/ld_r18_voc_loc_only.py --gpu-ids 0
   │
   ├─ ③ Config.fromfile()：{{ fileDirname }} 替换 → _base_ 递归合并 → 3 行覆盖
   ├─ ④ build_dataset / build_detector → 模型和教师都建出来
   ├─ ⑤ EpochBasedRunner 开始循环
   ▼
每个 iteration：
   x      = student.extract_feat(img)
   out_T  = teacher.bbox_head(teacher.extract_feat(img))     ← 同一个 img
   losses = LDHead.forward_train(x, out_T, gt_bboxes, gt_labels)
              └─ loss_single: loss_cls / loss_bbox / loss_dfl
                              / loss_ld / loss_ld_vlr / loss_kd
```

**讲稿**：

> 关键在**第二步**：官方入口只接受'一个 config 文件路径'，我没法从命令行告诉它'数据集在这、图缩到 640×384、batch 设成 1'。官方的侧门 `--cfg-options` 要改十几个嵌套的键，命令行会长到没法看。
>
> 所以我把这些参数塞进**环境变量**，config 在被读取的时候自己去 `os.environ.get()`。**环境变量是这层壳和官方代码之间唯一的接口**，一共 10 个。

---





