# BR-LiteDepth：原版训练结束后的逐文件实施手册

适用仓库：[muxinkexin/BR-LiteDepth](https://github.com/muxinkexin/BR-LiteDepth)。本手册核对提交：`26f43240c6fcee819ca73f001dd783b26c331580`，2026-11-01。

**本版给出完整替换代码，不再把 E1/E2/E3 留成“待实现接口”。不改现有目录框架：永久修改仅涉及 6 个已有文件；临时安装、检查、评估、启动脚本都在附录，放项目根目录即可，用完可以删除。**

本轮目标依次是：保留当前原版 E0 → 全量跨尺寸评价 → 加入固定画布增强 E1 → 普通一致性 E2 → 带宽一致性 E3 → 两个条件 CDC 的 E4/E5。暂不叠加新解码器、Mamba 或教师蒸馏。

本文没有向 GitHub 推送修改，也没有停止你的训练。代码在独立副本上做了 CPU 合成数据测试，包括前向/反向、单步优化、保存/严格重载与多尺寸评估；没有完成真实 KITTI/GPU 训练，不能把下述操作当作已取得性能提升。

## 1. 当前训练继续，先保存原版结果

### 1.1 正在运行时不要替换当前进程所用源码

让本轮 E0 正常结束。DataLoader worker 可能重新导入数据代码，不建议训练过程中覆盖文件。若要提前准备，可用独立 Git worktree；这只是另一份工作副本，内部目录框架不变。

```bash
# 只在需要提前准备独立副本时执行；目录名称可自定。
git worktree add -b research/resolution-v3 ../BR-LiteDepth-resolution 26f43240c6fcee819ca73f001dd783b26c331580
```

你现有环境适配保留：`train.py` 清理 LD 环境的代码、`checkpoint_utils.py` 的安全预训练加载、Pillow 新接口都不覆盖。预训练权重及数据用原来的绝对路径，无需再下载。

### 1.2 训练结束后设置一次变量

进入你准备操作的仓库根目录，把以下路径换成实际值。不要根据仓库默认值猜测此次训练实际参数，以该 run 的 `models/opt.json` 为准。

```bash
export BR_E0_RUN=/your/checkpoint/E0_pretrained_s0
export BR_E0_OPT="$BR_E0_RUN/models/opt.json"
export BR_E0_WEIGHTS="$BR_E0_RUN/models/weights_29"
export BR_DATA=/your/path/KITTI
export BR_RESULTS=/your/path/br_resolution_results
export BR_IMAGE_EXT=png
mkdir -p "$BR_RESULTS"
```

`weights_29` 只是训练 30 epoch 的例子。你仓库 `num_epochs` 默认是 50，实际是否传了 30 要看 opt.json。先运行：

```bash
python - "$BR_E0_OPT" <<'PY'
import json,sys
o=json.load(open(sys.argv[1]))
for k in ['model','data_path','png','mypretrain','load_weights_folder','height','width',
          'scales','batch_size','num_epochs','lr','drop_path','weight_decay','weights_init','log_dir']:
    print(k,':',o.get(k))
PY
```

确认训练结束、最终 checkpoint 包含 `encoder.pth/depth.pth`。同时保留 `opt.json`、console/TensorBoard 日志、当前提交与原始预训练权重。当前 run 名中的 `_s0` 不代表种子已设置：此提交的 `train.py` 还没有 seed 逻辑。

## 2. 不改原评估器，先评价 E0

把附录 C 保存为根目录 `br_evaluate.py`。它同时支持修改前的 E0 和修改后的 E1–E5，无需改变 `evaluate_depth.py`。

你提交中的评估器存在三个不适合本实验的默认行为：从 checkpoint 固定读入训练尺寸；评估数据集没有传 `--png`；每个 batch 都运行 profiling。附录脚本独立处理这些问题，不把它们混进算法修改。

### 2.1 测试集与 GT

仓库已包含 `splits/eigen/test_files.txt`、`splits/eigen/gt_depths.npz` 和 `eigen_zhou` 训练/验证 split，不重复导出覆盖它们。

```bash
wc -l splits/eigen_zhou/train_files.txt splits/eigen_zhou/val_files.txt splits/eigen/test_files.txt
test -f splits/eigen/gt_depths.npz
```

标准清单应是 39,810 / 4,424 / 697。图像格式从 E0 opt 的 `png` 确认，PNG 对应 `BR_IMAGE_EXT=png`，JPG 对应 `jpg`。GT 必须对应同一测试清单和顺序。

### 2.2 一条命令评价五个输入尺寸

```bash
python br_evaluate.py \
  --weights "$BR_E0_WEIGHTS" --data_path "$BR_DATA" --ext "$BR_IMAGE_EXT" \
  --split_file splits/eigen/test_files.txt --gt splits/eigen/gt_depths.npz \
  --out "$BR_RESULTS/E0_original" --device cuda
```

默认尺寸为 640×192、416×128、512×160、832×256、1024×320（宽×高）。网络只按 checkpoint 的**训练尺寸**构造一次，改变真实输入 tensor 尺寸，避免按测试尺寸重新选择 dilation。

输出：

| 文件 | 含义 |
|---|---|
| 每个尺寸目录中的 `disps_eigen_split.npy` | 物理逆深度，非 sigmoid 原始输出；不能再调用一次 disp_to_depth |
| `metadata.json`、`files.txt` | 权重 hash、训练/测试尺寸、配置、图像顺序 |
| 根输出目录 `summary.json` | 标准七指标、固定参考尺度七指标、尺度漂移、非参考尺寸平均 |
| `per_image.csv` | 每图、每尺寸的结果，用于配对比较 |

697 张的五尺寸 float32 预测约占 2.2 GB 磁盘；脚本用 memmap 写入，避免把全部预测堆在内存中。GT 解压仍需要内存。可先只用 `--sizes 640x192 416x128` 排查流程，最终再用新输出目录完整评价。

同一结果已经生成且配置一致时，重新统计可加 `--reuse_predictions`；脚本会校验 checkpoint hash、参数和清单。不允许悄悄覆盖旧结果，中断留下的半成品应换新输出目录重做。

### 2.3 指标协议

标准协议沿用当前代码：先把物理逆深度插值到原 GT 尺寸，再取倒数；相同 crop；GT 有效范围 1e-3–80m；每图 median scaling；不做翻转后处理。

另外两项必须单列：

- `shared_*`：每张图只用参考 640×192 预测求一次 GT median 比例，其他尺寸共用它；这是分辨率尺度稳定性评价，不是绝对尺度恢复。
- `drift`：未经独立缩放、未经 80m 截断的预测中，`abs(median(log Z_r - log Z_ref))`。参考尺寸应为 0。

标准 median scaling 会掩盖整体尺度漂移，不能只凭标准 AbsRel 宣称模型已适应分辨率。该脚本的像素对应仅用于**纯 resize**，后续 crop/变焦评估要另做几何对齐。

## 3. 这轮实验怎样分组

| ID | 视图 | 一致性 | 条件 CDC | 作用 |
|---|---|---|---|---|
| 当前 E0 | 单 M | 无 | 无 | 历史原版结果，继续保留 |
| E0R | 单 M | 无 | 无 | 统一 seed、Pose 模式修复后的单路控制 |
| EC | 相同 M 重复三路 | 无 | 无 | 控制额外前向、BN/DropPath 和严格 mask 的影响 |
| E1 | L/M/H | 无 | 无 | 固定画布尺度增强 |
| E2 | 与 E1 同分布 | 细网格深度 SSIM+L1 | 无 | 普通跨尺度对照 |
| E2log | 与 E1 同分布 | 粗网格 log-depth，不滤深度 | 无 | 与 E3 共享网格、有效区域、损失形式的关键对照 |
| E3 | 与 E1 同分布 | 粗网格 log-depth，滤波深度 | 无 | 单独验证低通/带宽收益 |
| E4 | 与 E2 相同 | 与 E2 相同 | 两块 | 普通一致性下检验条件卷积 |
| E5 | 与 E3 相同 | 与 E3 相同 | 两块 | 联合方案 |

优先级：先 E1→E2→E2log→E3，确认改善再 E4/E5；最终结论前补 E0R/EC。E3 只比 E2 好不能证明收益来自低通，因为损失形式也变了，必须比较 E2log。

本版 L 使用 masked padding，**不是 RA 原始复制拼接的完整复现**。论文阶段还应补原始 RA 策略迁移到 tiny 的参照；当前先把最小研究链跑通。

### 3.1 本次修复与历史 E0 的关系

当前 `set_eval()` 只切 DepthNet，PoseNet 在验证时仍可能处于 train，更新 BN。第 4 节给出同时切换两个网络的修复。训练、验证输出没有用深度标签反传，GT 仍仅用于监测。

当前 E0 无固定 seed、保留了上述行为，因此它是历史参考，不能单独与 E3 做严格增益归因。E1/E2/E2log/E3 都使用同一修复和初始化；E0R/EC 提供最终更严格对照。没有必要中断已经运行的 E0。

## 4. 永久代码修改：6 个已有文件，24 处操作

**以下内容全部应用后再训练，不能只注册参数就运行 E1。** 方法、类中的代码保留原缩进；“追加”与“替换”已标明。`checkpoint_utils.py`、`evaluate_depth.py`、原网络解码器和优化器实现不改，也不新建 losses/configs 等目录。

你可以逐处手动修改；也可将本 .md 放项目根目录，把附录 A 保存为 `br_apply_manual.py`，先 dry run 再应用：

```bash
python br_apply_manual.py --manual BR-LiteDepth_PostTraining_Manual.md
# 核对 dry run、确认 E0 已结束后，再执行：
python br_apply_manual.py --manual BR-LiteDepth_PostTraining_Manual.md --apply
git diff --stat
```

自动安装器先检查全部替换和语法，再为每个文件保存 `.br_v3.bak`，没有创建新源码目录。请使用下载的原始 .md 文件：安装器需要其中的隐藏注释标记，不要从渲染页面复制一份不含标记的文本。已有本版备份时拒绝重复应用。你本地与提交不一致时可能匹配失败，应核对差异，不删除断言强行跑。

### 4.1 `options.py`

**4.1.1 在 LiteMonoOptions.__init__ 的 # PATHS 前增加参数**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "options.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
        # PATHS
```

替换为：

```python
        self.parser.add_argument('--seed',type=int,default=0)
        self.parser.add_argument('--resolution_aug',choices=['none','pad3','repeat3'],default='none')
        self.parser.add_argument('--consistency',choices=['none','ra','log','band'],default='none')
        self.parser.add_argument('--lambda_cs',type=float,default=1.0)
        self.parser.add_argument('--lambda_band',type=float,default=0.2)
        self.parser.add_argument('--consistency_start_epoch',type=int,default=5)
        self.parser.add_argument('--consistency_ramp_epochs',type=int,default=5)
        self.parser.add_argument('--conditional_cdc',action='store_true')

        # PATHS
```
<!-- BR_EDIT_END -->

### 4.2 `train.py`

**4.2.1 保留环境清理代码，只替换最后的 trainer 构造行**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "train.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
    trainer = Trainer(opts)
```

替换为：

```python
    import random
    import numpy as np
    import torch
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    trainer = Trainer(opts)
```
<!-- BR_EDIT_END -->

### 4.3 `datasets/mono_dataset.py`

**4.3.1 在 MonoDataset.__init__ 增加默认开关**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "datasets/mono_dataset.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
        self.is_train = is_train
```

替换为：

```python
        self.is_train = is_train
        self.resolution_aug = 'none' 
```
<!-- BR_EDIT_END -->

**4.3.2 在 preprocess 首次遍历前保留原始 PIL 图**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "datasets/mono_dataset.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
        for k in list(inputs):
            frame = inputs[k]
```

替换为：

```python
        native = {fid: inputs[('color',fid,-1)] for fid in self.frame_idxs}
        for k in list(inputs):
            frame = inputs[k]
```
<!-- BR_EDIT_END -->

**4.3.3 在 preprocess 最后调用增强，插入到 def __len__ 前**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "datasets/mono_dataset.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
    def __len__(self):
```

替换为：

```python
        if self.is_train and self.resolution_aug != 'none':
            self._build_resolution_views(inputs,native,color_aug)

    def __len__(self):
```
<!-- BR_EDIT_END -->

**4.3.4 在 MonoDataset 类末尾追加方法，保持四空格缩进**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "datasets/mono_dataset.py", "kind": "append"} -->
追加以下完整代码：

```python
    def _build_resolution_views(self, inputs, native, color_aug):
        """Flat tensor dictionary: original keys remain the M-view interface."""
        h, w = self.height, self.width
        mode = self.resolution_aug
        inputs[('br_valid', 'M', 0)] = torch.ones(1,h,w)
        if mode == 'repeat3':
            specs = [('L',h,w,0,0),('H',h,w,0,0)]
        else:
            sl, sh = random.uniform(0.7,0.9), random.uniform(1.1,2.0)
            hl, wl = int(h*sl), int(w*sl)
            hh, wh = int(h*sh), int(w*sh)
            specs = [('L',hl,wl,0,0),
                     ('H',hh,wh,random.randint(0,wh-w),random.randint(0,hh-h))]
        for name, rh, rw, crop_x, crop_y in specs:
            mask = torch.zeros(1,h,w)
            mask[:, :min(rh,h), :min(rw,w)] = 1.0
            if name == 'H' or mode == 'repeat3':
                mask.fill_(1.0)
            inputs[('br_valid',name,0)] = mask
            sx, sy = rw/w, rh/h
            a = torch.eye(4,dtype=torch.float32)
            a[0,0], a[1,1] = sx, sy
            a[0,2] = (sx-1)/2-crop_x
            a[1,2] = (sy-1)/2-crop_y
            k0 = a @ inputs[('K',0)]
            for scale in range(self.num_scales):
                ds = 2**scale
                down = torch.eye(4)
                down[0,0] = down[1,1] = 1/ds
                down[0,2] = down[1,2] = (1/ds-1)/2
                kv = down @ k0
                inputs[('br_K',name,scale)] = kv
                inputs[('br_inv_K',name,scale)] = torch.linalg.inv(kv)
            for fid in self.frame_idxs:
                if mode == 'repeat3':
                    for scale in range(self.num_scales):
                        inputs[('br_color',name,fid,scale)] = inputs[('color',fid,scale)]
                        inputs[('br_color_aug',name,fid,scale)] = inputs[('color_aug',fid,scale)]
                    continue
                resized = native[fid].resize((rw,rh),resample=self.interp)
                if name == 'L':
                    image = Image.new('RGB',(w,h))
                    image.paste(resized,(0,0))
                else:
                    image = resized.crop((crop_x,crop_y,crop_x+w,crop_y+h))
                for scale in range(self.num_scales):
                    im = image if scale == 0 else self.resize[scale](image)
                    inputs[('br_color',name,fid,scale)] = self.to_tensor(im)
                    inputs[('br_color_aug',name,fid,scale)] = self.to_tensor(color_aug(im))
```
<!-- BR_EDIT_END -->

### 4.4 `layers.py`

**4.4.1 在 layers.py 末尾追加 BR 损失函数**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "layers.py", "kind": "append"} -->
追加以下完整代码：

```python
# BR-LiteDepth: tensor-only helpers; no new package/directory is required.
def br_masked_mean(x, mask):
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def br_erode(mask, radius=1):
    if radius == 0:
        return mask
    invalid = F.pad(1.0 - mask, (radius,) * 4, value=1.0)
    return 1.0 - F.max_pool2d(invalid, 2 * radius + 1, stride=1)


def br_smooth_loss(disp, image, mask):
    mean = (disp * mask).sum((2, 3), keepdim=True)
    mean = mean / mask.sum((2, 3), keepdim=True).clamp_min(1.0)
    d = disp / (mean + 1e-7)
    dx = (d[..., 1:] - d[..., :-1]).abs()
    dy = (d[..., 1:, :] - d[..., :-1, :]).abs()
    ix = (image[..., 1:] - image[..., :-1]).abs().mean(1, True)
    iy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(1, True)
    mx = mask[..., 1:] * mask[..., :-1]
    my = mask[..., 1:, :] * mask[..., :-1, :]
    return br_masked_mean(dx * torch.exp(-ix), mx) + br_masked_mean(dy * torch.exp(-iy), my)


def br_gaussian(x, sigma_x, sigma_y):
    # sigma is determined by known image geometry, not learned.
    for axis, sigma in [('x', sigma_x), ('y', sigma_y)]:
        if sigma < 1e-5:
            continue
        radius = int(math.ceil(3.0 * sigma))
        t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        w = torch.exp(-0.5 * (t / sigma).square())
        w = w / w.sum()
        kernel = w.view(1, 1, 1, -1) if axis == 'x' else w.view(1, 1, -1, 1)
        pad = (0, radius) if axis == 'x' else (radius, 0)
        x = F.conv2d(x, kernel, padding=pad)
    return x


def br_cross_scale(za, zb, ka, kb, ma, mb, mode):
    """Return loss and common-grid valid fraction for a batch.

    mode=ra: fine common grid, unfiltered SSIM+L1 depth control.
    mode=log: coarse common grid, unfiltered log-depth control.
    mode=band: SAME coarse grid/log loss as log, with Gaussian prefilter.
    """
    assert mode in ('ra', 'log', 'band')
    assert za.shape == zb.shape and za.shape[1] == 1
    losses, fractions = [], []
    h, w = za.shape[-2:]
    for n in range(za.shape[0]):
        aa, bb = za[n:n+1], zb[n:n+1]
        va, vb = ma[n:n+1], mb[n:n+1]
        ca, cb = ka[n, :3, :3], kb[n, :3, :3]
        fxa, fya, cxa, cya = [float(ca[i, j]) for i, j in [(0,0),(1,1),(0,2),(1,2)]]
        fxb, fyb, cxb, cyb = [float(cb[i, j]) for i, j in [(0,0),(1,1),(0,2),(1,2)]]
        assert min(fxa, fya, fxb, fyb) > 0
        choose = max if mode == 'ra' else min
        fx, fy = choose(fxa, fxb), choose(fya, fyb)
        # Find rectangular real-image support, before filtering.
        boxes = []
        for vm, fxx, fyy, cx, cy in [(va,fxa,fya,cxa,cya),(vb,fxb,fyb,cxb,cyb)]:
            yy, xx = torch.where(vm[0,0] > 0.5)
            if xx.numel() == 0:
                raise RuntimeError('BR consistency: empty real-image support')
            boxes.append(((float(xx.min())-cx)/fxx, (float(xx.max())-cx)/fxx,
                          (float(yy.min())-cy)/fyy, (float(yy.max())-cy)/fyy))
        left, right = max(b[0] for b in boxes), min(b[1] for b in boxes)
        top, bottom = max(b[2] for b in boxes), min(b[3] for b in boxes)
        nx = int(math.floor((right-left)*fx)) + 1
        ny = int(math.floor((bottom-top)*fy)) + 1
        if min(nx, ny) < 4:
            raise RuntimeError('BR consistency: insufficient overlapping field of view')
        xs = left + torch.arange(nx,device=za.device,dtype=za.dtype)/fx
        ys = top + torch.arange(ny,device=za.device,dtype=za.dtype)/fy
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        samples, supports = [], []
        for depth, valid, fxx, fyy, cx, cy in [(aa,va,fxa,fya,cxa,cya),(bb,vb,fxb,fyb,cxb,cyb)]:
            if mode in ('log','band'):
                sx = 0.5 * math.sqrt(max((fxx/fx)**2-1.0, 0.0))
                sy = 0.5 * math.sqrt(max((fyy/fy)**2-1.0, 0.0))
                weight = br_gaussian(valid, sx, sy)
                if mode == 'band':
                    depth = br_gaussian(depth*valid, sx, sy) / weight.clamp_min(1e-6)
                # log/band compare exactly the same conservative support.
                valid = (weight > 0.9999).to(depth.dtype)
            grid = torch.stack((2*(xx*fxx+cx)/(w-1)-1,
                                2*(yy*fyy+cy)/(h-1)-1), -1)[None]
            samples.append(F.grid_sample(depth,grid,mode='bilinear',padding_mode='zeros',align_corners=True))
            supports.append(F.grid_sample(valid,grid,mode='bilinear',padding_mode='zeros',align_corners=True))
        mask = ((supports[0] > 0.9999) & (supports[1] > 0.9999)).to(za.dtype)
        # Same erosion for all modes; SSIM has a 3x3 support.
        mask = br_erode(mask, 1)
        if float(mask.sum()) < 4:
            raise RuntimeError('BR consistency: no reliable common pixels')
        a, b = samples
        if mode == 'ra':
            error = 0.85 * SSIM()(a,b) + 0.15 * (a-b).abs()
        else:
            delta = a.clamp_min(1e-6).log() - b.clamp_min(1e-6).log()
            error = torch.sqrt(delta.square()+1e-6)
        losses.append(br_masked_mean(error,mask))
        fractions.append(mask.mean().detach())
    return torch.stack(losses).mean(), torch.stack(fractions).mean()
```
<!-- BR_EDIT_END -->

### 4.5 `trainer.py`

**4.5.1 在 Trainer.__init__ 明确本实现支持范围**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "trainer.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
        self.opt = options
```

替换为：

```python
        self.opt = options
        if self.opt.resolution_aug != 'none' or self.opt.conditional_cdc:
            assert self.opt.dataset == 'kitti'
            assert self.opt.pose_model_type == 'separate_resnet' and self.opt.pose_model_input == 'pairs'
            assert self.opt.frame_ids == [0,-1,1] and self.opt.scales == [0,1,2]
            assert not self.opt.use_stereo and not self.opt.v1_multiscale
            assert not self.opt.avg_reprojection and not self.opt.predictive_mask
        assert self.opt.consistency == 'none' or self.opt.resolution_aug != 'none'

```
<!-- BR_EDIT_END -->

**4.5.2 编码器构造增加条件开关**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "trainer.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
width=self.opt.width, height=self.opt.height)
```

替换为：

```python
width=self.opt.width, height=self.opt.height,
                                                   conditional_cdc=self.opt.conditional_cdc)
```
<!-- BR_EDIT_END -->

**4.5.3 仅对训练数据集打开增强**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "trainer.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
        self.train_loader = DataLoader(
```

替换为：

```python
        train_dataset.resolution_aug = self.opt.resolution_aug
        self.train_loader = DataLoader(
```
<!-- BR_EDIT_END -->

**4.5.4 process_batch 的 tensor 迁移后增加三路分支**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "trainer.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type
```

替换为：

```python
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if ('br_valid','M',0) in inputs:
            return self.process_resolution_batch(inputs)

        if self.opt.pose_model_type
```
<!-- BR_EDIT_END -->

**4.5.5 单路及验证前向传递当前 K**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "trainer.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
features = self.models["encoder"](inputs["color_aug", 0, 0])
```

替换为：

```python
features = self.models["encoder"](inputs["color_aug", 0, 0],camera=inputs[('K',0)])
```
<!-- BR_EDIT_END -->

**4.5.6 在 encoder checkpoint 记录推理配置**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "trainer.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
                to_save['use_stereo'] = self.opt.use_stereo
```

替换为：

```python
                to_save['use_stereo'] = self.opt.use_stereo
                to_save['model_name'] = self.opt.model
                to_save['conditional_cdc'] = self.opt.conditional_cdc
                to_save['min_depth'] = self.opt.min_depth
                to_save['max_depth'] = self.opt.max_depth
```
<!-- BR_EDIT_END -->

**4.5.7 在 Trainer 类末尾追加两个完整方法**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "trainer.py", "kind": "append"} -->
追加以下完整代码：

```python
    def process_resolution_batch(self, inputs):
        # M keeps original keys, so the original TensorBoard/depth monitoring works.
        views = {'M': inputs}
        for name in ('L', 'H'):
            v = {}
            for s in range(4):
                v[('K',s)] = inputs[('br_K',name,s)]
                v[('inv_K',s)] = inputs[('br_inv_K',name,s)]
                for fid in self.opt.frame_ids:
                    v[('color',fid,s)] = inputs[('br_color',name,fid,s)]
                    v[('color_aug',fid,s)] = inputs[('br_color_aug',name,fid,s)]
            views[name] = v
        poses = self.predict_poses(inputs, None)
        predictions, losses, native = {}, {}, []
        for name in ('M','L','H'):
            v = views[name]
            feat = self.models['encoder'](v[('color_aug',0,0)],camera=v[('K',0)])
            out = self.models['depth'](feat)
            out.update(poses)
            self.generate_images_pred(v,out)
            value = self.compute_resolution_native(v,out,inputs[('br_valid',name,0)])
            predictions[name] = out
            native.append(value)
            losses['br/native_'+name] = value
            losses['br/photo_valid_'+name] = torch.stack([out['br_valid_fraction/{}'.format(s)] for s in self.opt.scales]).mean()
        loss = sum(native)/len(native)
        consistency = loss*0.0
        if self.opt.consistency != 'none':
            pairs, coverage = [], []
            for a,b in [('L','M'),('M','H')]:
                x, fraction = br_cross_scale(
                    predictions[a][('depth',0,0)], predictions[b][('depth',0,0)],
                    views[a][('K',0)],views[b][('K',0)],
                    inputs[('br_valid',a,0)],inputs[('br_valid',b,0)],self.opt.consistency)
                pairs.append(x); coverage.append(fraction)
            consistency = sum(pairs)/len(pairs)
            losses['br/common_valid'] = sum(coverage)/len(coverage)
        ramp = min(max((self.epoch-self.opt.consistency_start_epoch+1)
                       /max(self.opt.consistency_ramp_epochs,1),0.0),1.0)
        weight = self.opt.lambda_cs if self.opt.consistency == 'ra' else self.opt.lambda_band
        losses['br/consistency'] = consistency
        losses['br/weight'] = loss.new_tensor(ramp*weight)
        losses['loss'] = loss + ramp*weight*consistency
        return predictions['M'], losses

    def compute_resolution_native(self, inputs, outputs, valid):
        # Same minimum identity/reprojection strategy as original, now masked.
        total = valid.sum()*0.0
        for scale in self.opt.scales:
            target = inputs[('color',0,0)]
            depth = outputs[('depth',0,scale)]
            points = self.backproject_depth[0](depth,inputs[('inv_K',0)])
            errors, identities, masks = [], [], []
            for fid in self.opt.frame_ids[1:]:
                grid = outputs[('sample',fid,scale)]
                source_valid = F.grid_sample(valid,grid,mode='bilinear',
                                             padding_mode='zeros',align_corners=True)
                in_bounds = (grid.abs() <= 1.0).all(-1).unsqueeze(1)
                z = (outputs[('cam_T_cam',0,fid)] @ points)[:,2:3]
                z = z.reshape_as(valid)
                mask = ((source_valid>0.9999) & (valid>0.5) & in_bounds & (z>1e-6)).float()
                mask = br_erode(mask,1)
                error = self.compute_reprojection_loss(outputs[('color',fid,scale)],target)
                identity = self.compute_reprojection_loss(inputs[('color',fid,0)],target)
                errors.append(error.masked_fill(mask<0.5,1e3))
                identities.append(identity.masked_fill(mask<0.5,1e3))
                masks.append(mask)
            valid_any = torch.stack(masks,0).amax(0)
            photo = torch.cat(errors,1)
            if not self.opt.disable_automasking:
                identity = torch.cat(identities,1)
                identity = identity + torch.randn_like(identity)*1e-5
                combined = torch.cat([identity,photo],1)
                values, indices = combined.min(1,keepdim=True)
                outputs['identity_selection/{}'.format(scale)] = (
                    (indices[:,0]>=len(errors)).float()*valid_any[:,0])
            else:
                values = photo.min(1,keepdim=True).values
            photo_loss = br_masked_mean(values,valid_any)
            disp = outputs[('disp',scale)]
            mask_small = F.interpolate(valid,size=disp.shape[-2:],mode='nearest')
            mask_small = br_erode(mask_small,1)
            smooth = br_smooth_loss(disp,inputs[('color',0,scale)],mask_small)
            total = total + photo_loss + self.opt.disparity_smoothness*smooth/(2**scale)
            outputs['br_valid_fraction/{}'.format(scale)] = valid_any.mean().detach()
        return total/len(self.opt.scales)
```
<!-- BR_EDIT_END -->

**4.5.8 在 set_train 中同时切换 PoseNet 模式（修复验证更新 Pose BN）**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "trainer.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
        for m in self.models.values():
            m.train()
```

替换为：

```python
        for m in list(self.models.values()) + list(self.models_pose.values()):
            m.train()
```
<!-- BR_EDIT_END -->

**4.5.9 在 set_eval 中同时切换 PoseNet 模式（修复验证更新 Pose BN）**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "trainer.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
        for m in self.models.values():
            m.eval()
```

替换为：

```python
        for m in list(self.models.values()) + list(self.models_pose.values()):
            m.eval()
```
<!-- BR_EDIT_END -->

### 4.6 `networks/depth_encoder.py`

**4.6.1 在 DilatedConv 前插入条件深度卷积类**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "networks/depth_encoder.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
class DilatedConv(nn.Module):
```

替换为：

```python
class BRConditionalDepthwise(nn.Module):
    """Preserve the existing .conv state_dict keys and share one kernel."""
    def __init__(self, old, reference_focal):
        super().__init__()
        self.conv = old.conv
        c = self.conv.in_channels
        self.gate = nn.Sequential(nn.Linear(c+4,16),nn.GELU(),nn.Linear(16,c))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias,-2.0)
        self.register_buffer('reference_focal',torch.tensor(reference_focal,dtype=torch.float32))

    def forward(self,x,camera):
        if camera is None:
            raise ValueError('conditional CDC requires current input K')
        k, image_hw = camera
        k = k.to(device=x.device,dtype=x.dtype)
        b,c,h,w = x.shape
        ih, iw = image_hw
        ratio = torch.stack([k[:,0,0],k[:,1,1]],1)/self.reference_focal.to(x.dtype)
        r = self.conv.dilation[0]
        d = (r*ratio).clamp(0.5,8.0)
        q = torch.cat([ratio.log(),(k[:,0,2]/iw-0.5)[:,None],
                        (k[:,1,2]/ih-0.5)[:,None]],1)
        g = torch.sigmoid(self.gate(torch.cat([x.mean((2,3)),q],1)))[:,:,None,None]
        yy,xx = torch.meshgrid(torch.arange(h,device=x.device,dtype=x.dtype),
                               torch.arange(w,device=x.device,dtype=x.dtype),indexing='ij')
        adaptive = torch.zeros_like(x)
        for ky in range(-1,2):
            for kx in range(-1,2):
                px = xx[None]+d[:,0,None,None]*kx
                py = yy[None]+d[:,1,None,None]*ky
                grid = torch.stack([2*px/max(w-1,1)-1,2*py/max(h-1,1)-1],-1)
                sample = F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=True)
                weight = self.conv.weight[:,0,ky+1,kx+1][None,:,None,None]
                adaptive = adaptive + sample*weight
        if self.conv.bias is not None:
            adaptive = adaptive + self.conv.bias[None,:,None,None]
        fixed = self.conv(x)
        return fixed + g*(adaptive-fixed)


class DilatedConv(nn.Module):
```
<!-- BR_EDIT_END -->

**4.6.2 仅替换 DilatedConv.forward 开头**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "networks/depth_encoder.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
    def forward(self, x):
        input = x

        x = self.ddwconv(x)
```

替换为：

```python
    def forward(self, x, camera=None):
        input = x

        x = self.ddwconv(x,camera) if isinstance(self.ddwconv,BRConditionalDepthwise) else self.ddwconv(x)
```
<!-- BR_EDIT_END -->

**4.6.3 LiteMono 构造参数增加开关**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "networks/depth_encoder.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):
```

替换为：

```python
heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], conditional_cdc=False, **kwargs):
```
<!-- BR_EDIT_END -->

**4.6.4 在原初始化之后包装两个 CDC，保留共享基座的随机初始化顺序**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "networks/depth_encoder.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
        self.apply(self._init_weights)

    def _init_weights
```

替换为：

```python
        self.apply(self._init_weights)
        self.conditional_cdc = conditional_cdc
        if conditional_cdc:
            assert model == 'lite-mono-tiny' and height == 192 and width == 640
            with torch.random.fork_rng(devices=[]):
                for i,j in [(1,2),(2,5)]:
                    old = self.stages[i][j].ddwconv
                    self.stages[i][j].ddwconv = BRConditionalDepthwise(old,(0.58*width,1.92*height))

    def _init_weights
```
<!-- BR_EDIT_END -->

**4.6.5 增加 stage 调用方法并显式传相机**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "networks/depth_encoder.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
    def forward_features(self, x):
```

替换为：

```python
    def _run_stage(self, index, x, camera):
        for block in self.stages[index]:
            x = block(x,camera) if isinstance(block,DilatedConv) else block(x)
        return x

    def forward_features(self, x, camera=None):
        camera = (camera,x.shape[-2:]) if camera is not None else None
```
<!-- BR_EDIT_END -->

**4.6.6 替换第一个 stage 的遍历**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "networks/depth_encoder.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
        for s in range(len(self.stages[0])-1):
            x = self.stages[0][s](x)
        x = self.stages[0][-1](x)
```

替换为：

```python
        x = self._run_stage(0,x,camera)
```
<!-- BR_EDIT_END -->

**4.6.7 替换后两个 stage 的遍历**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "networks/depth_encoder.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)
```

替换为：

```python
            x = self._run_stage(i,x,camera)
```
<!-- BR_EDIT_END -->

**4.6.8 替换 LiteMono.forward 的调用**

<!-- BR_EDIT_START -->
<!-- BR_EDIT_META {"file": "networks/depth_encoder.py", "kind": "replace"} -->
查找以下原片段（保留缩进，只匹配一次）：

```python
    def forward(self, x):
        x = self.forward_features(x)
```

替换为：

```python
    def forward(self, x, camera=None):
        x = self.forward_features(x,camera=camera)
```
<!-- BR_EDIT_END -->


## 5. 新代码实际实现了什么

### 5.1 数据与几何

所有训练 tensor 仍为 192×640。M 沿用原键名及原预处理；L 从原图按 0.7–0.9 倍 resize 后左上填充；H 从原图按 1.1–2.0 倍 resize 后裁剪。三帧共用同一几何参数，从原始 PIL 图直接生成，不把已经缩成 M 的图重新放大。

`br_K` 使用实际整数宽高和像素中心变换；有效 mask 排除 padding、投影越界、源端无效区及负 z。三路共用 M 上预测的 Pose；各路使用自身 K 做重投影。原始颜色参与光度监督，颜色增强图参与网络输入。

键保持为**扁平 tensor 字典**，原 `inputs[key].to(device)` 可用，无需改成嵌套目录或新的数据容器。验证集不启用三视图。

普通颜色增强沿用当前行为，本轮没有顺带重写其随机机制。若后续改成严格同步颜色参数，要统一重跑相关控制组。

### 5.2 Native loss 与普通一致性

三路使用 minimum identity/reprojection 的原思路，增加有效区域掩码；平滑项只比较两个端点都有效的邻居。三路 native loss 求平均，原 `[0,1,2]` 多输出监督保留。一致性只作用于最高分辨率深度，L↔M、M↔H 两对求平均。

E2 是普通细网格 `0.85*SSIM + 0.15*L1(depth)` 控制组，不声称逐字节复现 RA 原实现。E2log 和 E3 使用同一粗网格、同一保守有效区域与 Charbonnier log-depth，仅 E3 对深度先滤波。这样低通效果可单独归因。

E3 的 Gaussian 是可分离近似低通，sigma 根据两视图像素焦距比确定；不是严格的频谱截断，也不是尺度等变性定理。只有普通 resize/crop/pad 的已知几何，没有把深度按图像缩放比例乘除。Charbonnier 中加入了 1e-6，因此完全一致时 loss 下限约为 0.001，而非 0。

### 5.3 条件 CDC 与紧凑性

可选 `--conditional_cdc` 只包裹 tiny 的 `stages[1][2]`、`stages[2][5]`，分别处于 H/8、H/16。原 `.ddwconv.conv` 权重键保留，固定和连续采样分支共享核；没有复制完整编码器。

本仓库直接参数计数（只包含 DepthNet）：原版 **2,147,659**；条件版 **2,154,155**，增加 **6,496**。这是 CPU 构造后的参数数目，不是速度或精度承诺。连续采样含九次 grid_sample，可能增加实际延迟，必须测量。

条件包装使用 `fork_rng`，避免新增 gate 消耗随机数后改变后续解码器初始权重。推理相机信息由当前测试尺寸 K 提供；参考焦距对应当前仓库的归一化 K。若以后换准确标定或参考训练尺寸，必须统一改参照和对照协议。

## 6. 先执行可删除的检查脚本

把附录 B 保存为根目录 `br_check.py`：

```bash
python -m py_compile options.py train.py trainer.py layers.py datasets/mono_dataset.py networks/depth_encoder.py
python br_check.py
python br_check.py --pipeline
```

检查脚本只使用合成数据、CPU，不读 KITTI 或真实预训练权重；`--pipeline` 检查真实的三路 forward/loss/backward。GPU、真实数据吞吐和长训练稳定性仍需你服务器上的 smoke run。

预期 PASS 包括：解析平面/尺度偏移、mask、条件 CDC 在参考 K 下退化到固定卷积、batch 与单样本一致、梯度、共同初始化、repeat3、四种 loss 模式的三路反向。

mask 测试是在固定预测时仅修改损失输入无效值；不要求“改图像 padding 再重新前向后预测不变”，卷积感受野会跨过边界。

## 7. 训练命令：自动继承你这次 E0 的真实配置

把附录 D 保存为根目录 `br_train_next.py`。它读取实际 E0 `models/opt.json`，保留数据路径、PNG/JPG、预训练文件、batch、epoch、LR、drop_path、优化器参数等；只设置实验开关、seed 和唯一 run 名。

它默认 dry run，不启动训练；加 `--run` 才运行。与原 `train.py` 共用入口，保留你的运行环境清理。脚本只支持 E0 为 ImageNet 初始化、192×640、tiny 的本手册协议，若 E0 实际是深度 checkpoint 微调会明确拒绝，不能混用实验性质。

### 7.1 E1：先诊断，再完整训练

```bash
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E1 --seed 0 --smoke
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E1 --seed 0 --smoke --run
```

检查命令及新 run 的 `models/opt.json`。输出名为 `E1_pretrained_s0_smoke`，完整 console 写入同一 run 的 `console.log`。确认：无 NaN/Inf；L/M/H native loss 有限；`br/photo_valid_*` 非零且合理；图像/重投影正确；checkpoint 能评估。

然后从共同 ImageNet 初始化**重新**进行完整训练：

```bash
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E1 --seed 0 --run
```

输出 `E1_pretrained_s0`，继承 E0 的实际 epoch 数，不假定一定 30。不能从 smoke 最终 checkpoint 续训主结果。若同名 run 已存在，脚本会停止，避免覆盖实验。

### 7.2 E2 / E2log / E3

分别执行 smoke 和正式训练；只有验证前一阶段数据/梯度无问题后才全量跑：

```bash
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E2 --seed 0 --smoke --run
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E2 --seed 0 --run

python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E2log --seed 0 --smoke --run
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E2log --seed 0 --run

python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E3 --seed 0 --smoke --run
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E3 --seed 0 --run
```

E2 初始 λ_CS=1；E2log/E3 初始 λ_band=0.2。前 5 epoch 一致性权重为 0，第 6–10 epoch 线性增加。一轮 smoke 会计算一致性，但总损失中的权重仍为 0，不能用 smoke 的精度判断方法是否有效；附录检查会在热身后状态验证反向。

权重只作为起点。正式比较在验证集扫描，例如 E2 λ_CS∈{0.1,0.5,1.0}，E2log/E3 使用共同 λ_band∈{0.05,0.1,0.2}。附录 D 支持 `--lambda_cs`、`--lambda_band` 和 `--name`，无需为调参再改训练源码。例如：

```bash
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E2 --seed 0 \
  --lambda_cs 0.5 --name E2_cs05_s0 --run
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E3 --seed 0 \
  --lambda_band 0.1 --name E3_band01_s0 --run
```

失败的 smoke 修复后可用 `--name E3_smoke_retry1` 重新启动，保留旧日志。不要反复覆盖旧 run 或根据测试集选参。

### 7.3 E4 / E5

E3 相对 E2log 显示稳定收益后，再执行：

```bash
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E4 --seed 0 --smoke --run
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E4 --seed 0 --run
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E5 --seed 0 --smoke --run
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E5 --seed 0 --run
```

两者仍从同一 ImageNet 初始化，不能让 E5 从训练完的 E3 接着训，再与从头训练的 E4 比较。评估时附录 C 自动读取 checkpoint 的条件开关与深度范围，并 strict 加载新参数，防止新 gate 被静默丢弃。

### 7.4 控制组与多 seed

```bash
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage E0R --seed 0 --run
python br_train_next.py --reference_opt "$BR_E0_OPT" --stage EC --seed 0 --run
```

在 seed=0 筛选稳定方案后，关键成对组补 seed=1、2。CUDA 的 grid_sample backward 可能仍存在非位级确定性，seed 不能代替多次实验。每个完整实验看相同原始片段曝光次数/更新步数；三视图比单视图计算更多，另报 GPU 小时，不能称为完全相同计算预算。

如果三路显存不够，先统一确定 batch。只有某一组降 batch 会影响 BN，梯度累积不能完全消除这项差异。最严谨的主表应使用相同物理 batch；已运行 E0 仍可保留为历史参考。

## 8. 后续各 run 的评估完全共用一条命令

把新 run 的最终权重目录指定给 `--weights`。以 30 epoch 的 E3 为例（实际 epoch 从其 opt.json 确认）：

```bash
export BR_E3_WEIGHTS=/your/checkpoint/E3_pretrained_s0/models/weights_29
python br_evaluate.py \
  --weights "$BR_E3_WEIGHTS" --data_path "$BR_DATA" --ext "$BR_IMAGE_EXT" \
  --split_file splits/eigen/test_files.txt --gt splits/eigen/gt_depths.npz \
  --out "$BR_RESULTS/E3_pretrained_s0" --device cuda
```

E1/E2/E2log/E4/E5 仅改权重和输出目录。保持清单、编码、crop、深度截断和后处理一致；不要为每个测试尺寸挑不同 checkpoint。

### 8.1 验证集与 checkpoint 选择

调参使用 `eigen_zhou/val_files.txt`；为该清单另生成匹配的验证 GT 后，通过 `--split_file`、`--gt` 指定。附录 C 可接收任意同格式清单。不能用测试集 697 张选权重/λ。

若暂时没有验证 GT，第一轮各组统一最后 epoch，测试集只做固定协议评价，暂不大量调参。现有 `trainer.val()` 只看一个 minibatch，不是全量验证，不等于本手册的正式指标。

### 8.2 独立效率测量

```bash
python br_evaluate.py --weights "$BR_E3_WEIGHTS" \
  --out "$BR_RESULTS/E3_benchmark" --device cuda --benchmark_only
```

该模式不读图像/GT：FP32、batch=1、50 次预热、200 次测量，输出 `benchmark.json` 中的 median/p90、参数和峰值分配显存。统一设备和后台负载。参数更少不代表 grid_sample 版本一定更快。

### 8.3 主表与判断

| Run | Params | 640 AbsRel | 416 | 512 | 832 | 1024 | 非参考平均 | shared 平均 | drift | 640延迟 | GPU小时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 / E0R | | | | | | | | | | | |
| EC / E1 | | | | | | | | | | | |
| E2 / E2log | | | | | | | | | | | |
| E3 | | | | | | | | | | | |
| E4 / E5 | | | | | | | | | | | |

E3 至少应相对 E2log 改善非参考平均误差，参考尺寸没有明显牺牲，shared/drift 与局部细节不反向恶化，才继续扩大研究。可预先把参考 AbsRel 恶化超过绝对值 0.002 标记为显著权衡；这是工程筛选标准，不是预测结果。

固定验证样例做局部放大和统一色标，检查细杆、近车、远物体和弱纹理。KITTI 原始稀疏 LiDAR 不能直接当稠密边界真值计算全图 F1，RGB 纹理边缘也不能代替深度边界标签。

## 9. 必须记录和常见问题

保存源码提交、原始 E0 opt、预训练 hash、每个 run 的 opt/br_launch.json/console、checkpoint hash、逐图指标、GPU小时、seed 和有效区域比例。对 E2log/E3，common_valid 应在同一增强下相同；不要用比较范围缩小解释为精度改善。

| 现象 | 首先定位 |
|---|---|
| 新 flag 不认识 | 没有完成 options.py 修改，或执行了另一份项目的 train.py |
| L loss 很小但图像明显不对 | mask 分母/有效比例、padding 是否进入监督 |
| 一致性 loss 降、预测变平 | λ 过大、低通/对齐错误、native loss 被稀释 |
| 参考 K 条件卷积不等价 | 零填充、grid 归一化、align_corners、权重共享 |
| 条件版严格加载失败 | 评估源码没有对应类，或 conditional 配置/权重不一致 |
| 更改测试尺寸结果仍相同 | 真实 tensor shape；旧评估器仍固定训练尺寸 |
| Pose 监测指标波动 | 使用全量独立评估；确认两个网络的 train/eval 模式 |
| 旧 run 下再次启动训练 | 启动器应拒绝；不要覆盖历史 checkpoint |

本版没有把 `load_weights_folder` 变成严格 resume。原代码尚未完整恢复 epoch/scheduler/RNG，不能把重新加载模型说成连续训练。长任务中断恢复应另实现完整训练状态保存/恢复并做数值检查；当前主协议每组从共同预训练开始，避免混入旧 optimizer。

## 10. 临时文件何时可以删除

| 文件 | 可删除时机 |
|---|---|
| `br_apply_manual.py` | 安装完成、git diff/编译检查通过之后 |
| `br_check.py` | 检查通过后可删；改损失/CDC 时建议留到再次验证 |
| `br_evaluate.py` | 完成相关评估与速度测试后可删；保留源码版本/本手册以复现 |
| `br_train_next.py` | 所有通过它启动的训练进程结束后可删；不要在运行时删除 |
| `*.br_v3.bak` | 修改提交到 Git、结果核对并确认无需回退后 |

它们都不作为模型依赖被永久 import。不要删除 6 个源码中的训练/条件卷积实现，否则条件 checkpoint 无法正确推理。保留 opt、日志、summary/per_image 与权重；预测 .npy 完成统计后可按磁盘需要删除，但删除前先确认有可复现来源。

## 附录 A：临时安装器 br_apply_manual.py

保存到项目根目录。默认 dry run；`--apply` 才写文件。解析的是本 .md 中的原文替换块，不需要另建 patch/config 目录。

```python
"""Temporary patch installer. Default is a dry run; --apply writes local files."""
import argparse,ast,json,re,shutil
from pathlib import Path

ALLOWED={'options.py','train.py','trainer.py','layers.py',
         'datasets/mono_dataset.py','networks/depth_encoder.py'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manual',required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    text=Path(a.manual).read_text()
    blocks=re.findall(r'^<!-- BR_EDIT_START -->\n(.*?)^<!-- BR_EDIT_END -->$',text,re.M|re.S)
    assert len(blocks)==24,f'Expected 24 edits, found {len(blocks)}'
    originals={};changed={}
    for number,block in enumerate(blocks,1):
        meta=re.search(r'^<!-- BR_EDIT_META (.*?) -->$',block,re.M)
        assert meta,number
        data=json.loads(meta.group(1));file=data['file'];assert file in ALLOWED,file
        path=Path(file);assert path.is_file(),f'Run inside repository root: {file}'
        if file not in originals:
            assert not Path(file+'.br_v3.bak').exists(),f'Backup exists; do not apply twice: {file}'
            originals[file]=path.read_text();changed[file]=originals[file]
        snippets=re.findall(r'```python\n(.*?)\n```',block,re.S)
        if data['kind']=='append':
            assert len(snippets)==1
            changed[file]=changed[file].rstrip()+'\n\n'+snippets[0].rstrip()+'\n'
        else:
            assert len(snippets)==2
            old,new=snippets
            assert changed[file].count(old)==1,f'Edit {number} does not match uniquely: {file}'
            changed[file]=changed[file].replace(old,new,1)
    assert set(changed)==ALLOWED
    for file,code in changed.items():
        ast.parse(code,filename=file)
        print(file,'OK',len(code.splitlines()),'lines')
    if not a.apply:
        print('Dry run passed. Review the manual and add --apply after E0 finishes.');return
    for file in sorted(changed):
        path=Path(file);shutil.copy2(path,Path(file+'.br_v3.bak'))
        path.write_text(changed[file])
    print('Applied 24 edits to 6 existing files; no new source directory created.')

if __name__=='__main__':main()
```

## 附录 B：临时检查脚本 br_check.py

```python
"""Temporary root-level CPU checks; no KITTI/real weights are needed."""
import os,sys
if __name__ == '__main__' and ('LD_LIBRARY_PATH' in os.environ or 'LD_PRELOAD' in os.environ):
    env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
    os.execve(sys.executable,[sys.executable]+sys.argv,env)
import argparse,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import networks
from networks.depth_encoder import CDilated,BRConditionalDepthwise
from datasets.mono_dataset import MonoDataset
from layers import br_cross_scale,br_masked_mean,br_erode,SSIM,BackprojectDepth,Project3D

def fake_k(h,w):
    return torch.tensor([[.58*w,0,.5*w,0],[0,1.92*h,.5*h,0],[0,0,1,0],[0,0,0,1]],dtype=torch.float32)

class FakeDataset(MonoDataset):
    def __init__(self,mode,h=96,w=320):
        super().__init__('', ['synthetic 1 l'],h,w,[0,-1,1],4,is_train=True)
        self.K=np.array([[.58,0,.5,0],[0,1.92,.5,0],[0,0,1,0],[0,0,0,1]],dtype=np.float32)
        self.resolution_aug=mode
    def check_depth(self):return False
    def get_color(self,folder,frame_index,side,do_flip):
        y,x=np.mgrid[:180,:600]
        im=np.stack([(x+frame_index*3)%256,(y*2)%256,(x//3+y)%256],-1).astype(np.uint8)
        im=Image.fromarray(im)
        return im.transpose(getattr(Image,'Transpose',Image).FLIP_LEFT_RIGHT) if do_flip else im

def basic():
    k=fake_k(64,128)[None]
    yy,xx=torch.meshgrid(torch.arange(64),torch.arange(128),indexing='ij')
    z=(5+.1*(xx-k[0,0,2])/k[0,0,0]+.2*(yy-k[0,1,2])/k[0,1,1])[None,None]
    z.requires_grad_();q=z.detach().clone().requires_grad_();v=torch.ones_like(z)
    loss,_=br_cross_scale(z,q,k,k,v,v,'band')
    assert abs(float(loss)-.001)<1e-5
    shifted,_=br_cross_scale(z,q*1.2,k,k,v,v,'band')
    assert float(shifted)>.1
    shifted.backward();assert z.grad is not None and q.grad is not None
    a=torch.eye(4)[None];a[:,0,0]=a[:,1,1]=1.5;a[:,0,2]=.25-32;a[:,1,2]=.25-16
    kh=a@k
    zh=(5+.1*(xx-kh[0,0,2])/kh[0,0,0]+.2*(yy-kh[0,1,2])/kh[0,1,1])[None,None]
    plane,_=br_cross_scale(z.detach(),zh,k,kh,v,v,'band')
    assert float(plane)<.002,float(plane)
    m=torch.zeros_like(z);m[...,10:50,20:100]=1
    x=torch.rand_like(z);old=br_masked_mean(x,br_erode(m))
    x=x+(1-m)*10000
    assert torch.allclose(old,br_masked_mean(x,br_erode(m)))
    # Identical reference camera must recover the original convolution, including edges.
    torch.manual_seed(0);old=CDilated(8,8,3,d=3,groups=8)
    module=BRConditionalDepthwise(old,(.58*640,1.92*192))
    f=torch.rand(2,8,12,40);camera=fake_k(192,640)[None].repeat(2,1,1)
    reference=module(f,(camera,(192,640)))
    assert (reference-old(f)).abs().max()<1e-4
    camera[1,:2,:]*=1.5
    both=module(f,(camera,(192,640)))
    parts=torch.cat([module(f[i:i+1],(camera[i:i+1],(192,640))) for i in range(2)])
    assert torch.allclose(both,parts,atol=1e-5)
    both.square().mean().backward()
    assert module.conv.weight.grad is not None and module.gate[-1].weight.grad is not None
    # Optional modules must not change base/decoder random initialization.
    for conditional in [False,True]:
        torch.manual_seed(0)
        e=networks.LiteMono(model='lite-mono-tiny',height=192,width=640,conditional_cdc=conditional)
        d=networks.DepthDecoder(e.num_ch_enc,scales=[0,1,2])
        if not conditional:base_e=e.state_dict();base_d=d.state_dict()
        else:
            for name,value in base_e.items():assert torch.equal(value,e.state_dict()[name]),name
            for name,value in base_d.items():assert torch.equal(value,d.state_dict()[name]),name
    print('PASS: geometry, scale sensitivity, masks, CDC reference/batch/backward, common initialization')

def pipeline():
    from trainer import Trainer
    from options import LiteMonoOptions
    opt=LiteMonoOptions().parser.parse_args(['--no_cuda','--height','96','--width','320',
                                           '--batch_size','1','--num_workers','0'])
    # Bypass I/O-heavy constructor; exercise the real batch/loss implementation.
    t=Trainer.__new__(Trainer);t.opt=opt;t.device=torch.device('cpu');t.epoch=9
    t.ssim=SSIM();t.num_scales=3
    t.backproject_depth={0:BackprojectDepth(1,96,320)}
    t.project_3d={0:Project3D(1,96,320)}
    for conditional in [False,True]:
        e=networks.LiteMono(model='lite-mono-tiny',height=192,width=640,conditional_cdc=conditional)
        d=networks.DepthDecoder(e.num_ch_enc,scales=[0,1,2]);t.models={'encoder':e,'depth':d}
        shift=torch.nn.Parameter(torch.tensor(.01))
        def poses(inputs,features):
            result={}
            for fid in [-1,1]:
                mat=torch.eye(4)[None].clone();mat[:,0,3]=shift*fid
                result[('cam_T_cam',0,fid)]=mat
            return result
        t.predict_poses=poses
        for mode in ['none','ra','log','band']:
            opt.consistency=mode
            batch=next(iter(DataLoader(FakeDataset('pad3'),batch_size=1,num_workers=0)))
            out,loss=t.process_batch(batch)
            assert torch.isfinite(loss['loss']),mode
            loss['loss'].backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in e.parameters())
            print('PASS pipeline',conditional,mode,float(loss['loss'].detach()),
                  {k:float(v.detach()) for k,v in loss.items() if 'valid' in k})
            e.zero_grad(set_to_none=True);d.zero_grad(set_to_none=True)
    # Test frame/K alignment in the repeated-view control.
    item=FakeDataset('repeat3')[0]
    for name in ['L','H']:
        assert torch.equal(item[('color',0,0)],item[('br_color',name,0,0)])
        assert torch.allclose(item[('K',0)],item[('br_K',name,0)])
    print('PASS: repeat3 control')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pipeline',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    basic()
    if a.pipeline:pipeline()
```

## 附录 C：临时多尺寸评估/尺度漂移/效率脚本 br_evaluate.py

```python
"""Temporary root-level evaluator for the BR-LiteDepth manual."""
import os,sys,tempfile
if __name__ == '__main__':
    if 'LD_LIBRARY_PATH' in os.environ or 'LD_PRELOAD' in os.environ:
        env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
        os.execve(sys.executable,[sys.executable]+sys.argv,env)
    os.environ.setdefault('OMP_NUM_THREADS','4')
    os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),'litemono-matplotlib'))
import argparse,csv,hashlib,json,time,inspect
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import networks
from layers import disp_to_depth

SIZES=['640x192','416x128','512x160','832x256','1024x320']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def metric(g,z):
    q=np.maximum(g/z,z/g)
    return [np.mean(abs(g-z)/g),np.mean((g-z)**2/g),np.sqrt(np.mean((g-z)**2)),
            np.sqrt(np.mean((np.log(g)-np.log(z))**2)),
            np.mean(q<1.25),np.mean(q<1.25**2),np.mean(q<1.25**3)]

def report(out,sizes,gt_path):
    gt=np.load(gt_path,allow_pickle=True)['data']
    predictions={s:np.load(out/s/'disps_eigen_split.npy',mmap_mode='r') for s in sizes}
    count=len((out/'640x192'/'files.txt').read_text().splitlines())
    assert len(gt)==count and all(len(v)==count for v in predictions.values())
    columns=['abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3']
    rows=[]
    for i,g0 in enumerate(gt):
        h,w=g0.shape; mask=(g0>1e-3)&(g0<80)
        c=np.array([.40810811*h,.99189189*h,.03594771*w,.96405229*w]).astype(int)
        crop=np.zeros_like(mask);crop[c[0]:c[1],c[2]:c[3]]=True;mask &= crop
        if not mask.any():raise ValueError(f'GT image {i} has no valid crop pixels')
        g=g0[mask].astype(np.float64);zs={}
        for s,v in predictions.items():
            inv=cv2.resize(np.asarray(v[i]),(w,h),interpolation=cv2.INTER_LINEAR)
            assert np.isfinite(inv).all() and (inv>0).all()
            zs[s]=(1.0/inv[mask]).astype(np.float64)
        zr=zs['640x192'];alpha=np.median(g)/np.median(zr)
        for s,z in zs.items():
            own=np.median(g)/np.median(z)
            row=dict(image_index=i,resolution=s,
                     drift=float(abs(np.median(np.log(z)-np.log(zr)))),reference_alpha=float(alpha))
            for tag,value in [('standard',own),('shared',alpha)]:
                row.update({tag+'_'+k:float(v) for k,v in zip(columns,metric(g,np.clip(z*value,1e-3,80)))})
            rows.append(row)
    with (out/'per_image.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    fields=['drift']+['standard_'+k for k in columns]+['shared_'+k for k in columns]
    summary={s:{k:float(np.mean([r[k] for r in rows if r['resolution']==s])) for k in fields} for s in sizes}
    others=[s for s in sizes if s!='640x192']
    if others:summary['nonreference_mean']={k:float(np.mean([summary[s][k] for s in others])) for k in fields}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--data_path')
    p.add_argument('--ext',choices=['png','jpg'],default='png')
    p.add_argument('--split_file',default='splits/eigen/test_files.txt')
    p.add_argument('--gt',default='splits/eigen/gt_depths.npz')
    p.add_argument('--out',required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--sizes',nargs='+',default=SIZES)
    p.add_argument('--reuse_predictions',action='store_true')
    p.add_argument('--benchmark_only',action='store_true')
    p.add_argument('--no_gt',action='store_true')
    p.add_argument('--torch_threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.torch_threads);cv2.setNumThreads(0)
    device=torch.device(a.device)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    wd=Path(a.weights);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    enc=torch.load(wd/'encoder.pth',map_location='cpu',weights_only=True)
    dec=torch.load(wd/'depth.pth',map_location='cpu',weights_only=True)
    op=wd.parent/'opt.json';opt=json.loads(op.read_text()) if op.is_file() else {}
    model_name=enc.get('model_name',opt.get('model','lite-mono-tiny'))
    conditional=bool(enc.get('conditional_cdc',opt.get('conditional_cdc',False)))
    dh,dw=int(enc['height']),int(enc['width'])
    lo=float(enc.get('min_depth',opt.get('min_depth',0.1)))
    hi=float(enc.get('max_depth',opt.get('max_depth',100.0)))
    scales=opt.get('scales',[0,1,2]);assert list(scales)==[0,1,2]
    encoder=networks.LiteMono(model=model_name,height=dh,width=dw,conditional_cdc=conditional).to(device).eval()
    decoder=networks.DepthDecoder(encoder.num_ch_enc,scales=scales).to(device).eval()
    known={'height','width','use_stereo','model_name','conditional_cdc','min_depth','max_depth'}
    encoder.load_state_dict({k:v for k,v in enc.items() if k not in known},strict=True)
    decoder.load_state_dict(dec,strict=True)
    param_count=sum(p.numel() for p in encoder.parameters())+sum(p.numel() for p in decoder.parameters())
    camera_supported = 'camera' in inspect.signature(encoder.forward).parameters
    def predict(x,k):
        return decoder(encoder(x,camera=k) if camera_supported else encoder(x))
    sizes=list(dict.fromkeys(a.sizes))
    if not a.benchmark_only and '640x192' not in sizes:raise ValueError('Need reference 640x192 for scale stability')
    rows=None
    if not a.benchmark_only:
        if not a.data_path:raise ValueError('--data_path required')
        rows=Path(a.split_file).read_text().splitlines();assert rows
    benchmark={}
    for size in sizes:
        w,h=map(int,size.split('x'));assert min(h,w)>0 and h%32==0 and w%32==0
        k=torch.tensor([[.58*w,0,.5*w,0],[0,1.92*h,.5*h,0],[0,0,1,0],[0,0,0,1]],device=device)[None]
        target=out/size;target.mkdir(exist_ok=True)
        meta=dict(encoder_sha256=sha(wd/'encoder.pth'),decoder_sha256=sha(wd/'depth.pth'),
                  train_hw=[dh,dw],eval_hw=[h,w],model=model_name,conditional_cdc=conditional,
                  min_depth=lo,max_depth=hi,image_ext=a.ext,params=param_count,
                  split_sha256=sha(a.split_file) if rows is not None else None,post_process=False)
        dest=target/'disps_eigen_split.npy';meta_path=target/'metadata.json'
        if a.benchmark_only:
            x=torch.rand(1,3,h,w,device=device)
            with torch.inference_mode():
                for _ in range(50):predict(x,k)
                if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                times=[]
                for _ in range(200):
                    if device.type=='cuda':
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();predict(x,k);end.record();end.synchronize()
                        times.append(start.elapsed_time(end))
                    else:
                        t=time.perf_counter();predict(x,k);times.append((time.perf_counter()-t)*1000)
            benchmark[size]=dict(median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),
                                 params=param_count,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
            continue
        if a.reuse_predictions:
            assert dest.is_file() and meta_path.is_file()
            assert json.loads(meta_path.read_text())==meta,'Cached prediction configuration differs'
            assert (target/'files.txt').read_text()=='\n'.join(rows)+'\n'
            assert np.load(dest,mmap_mode='r').shape==(len(rows),h,w)
            continue
        if dest.exists():raise FileExistsError(f'{dest}: use a new output folder or --reuse_predictions')
        memory=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(rows),h,w))
        with torch.inference_mode():
            for i,row in enumerate(rows):
                folder,frame,side=row.split();side={'l':2,'r':3,'2':2,'3':3}[side]
                path=Path(a.data_path)/folder/f'image_0{side}'/'data'/f'{int(frame):010d}.{a.ext}'
                with Image.open(path) as im:
                    im=im.convert('RGB').resize((w,h),resample=getattr(Image,'Resampling',Image).LANCZOS)
                    arr=np.array(im,dtype=np.float32)/255.0
                x=torch.from_numpy(arr).permute(2,0,1)[None].to(device)
                raw=predict(x,k)[('disp',0)]
                assert tuple(raw.shape)==(1,1,h,w)
                inverse,_=disp_to_depth(raw,lo,hi)
                memory[i]=inverse[0,0].cpu().numpy()
                if i==0:print('actual tensor',tuple(x.shape),'train constructor',(dh,dw),'conditional',conditional)
        memory.flush();del memory
        # Commit metadata last: an interrupted file is not a valid reusable result.
        (target/'files.txt').write_text('\n'.join(rows)+'\n')
        meta_path.write_text(json.dumps(meta,indent=2))
    if a.benchmark_only:
        (out/'benchmark.json').write_text(json.dumps(benchmark,indent=2));print(json.dumps(benchmark,indent=2))
    elif not a.no_gt:report(out,sizes,a.gt)

if __name__=='__main__':main()
```

## 附录 D：临时训练启动器 br_train_next.py

```python
"""Temporary launcher: inherit the actual E0 training options, no new config tree."""
import argparse,hashlib,json,shlex,subprocess,sys
from pathlib import Path
from options import LiteMonoOptions

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference_opt',required=True)
    p.add_argument('--stage',choices=['E0R','EC','E1','E2','E2log','E3','E4','E5'],required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--name')
    p.add_argument('--lambda_cs',type=float)
    p.add_argument('--lambda_band',type=float)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    path=Path(a.reference_opt);original=json.loads(path.read_text())
    assert original.get('model')=='lite-mono-tiny'
    assert (original['height'],original['width'])==(192,640)
    assert original.get('mypretrain'),'E0 must use a common ImageNet pretrain file'
    assert not original.get('load_weights_folder'),'Fine-tuning E0 needs a separate matched protocol'
    assert Path(original['mypretrain']).expanduser().is_file(),'Pretrain path is not available in this working copy'
    modes={'E0R':('none','none',False),'EC':('repeat3','none',False),
           'E1':('pad3','none',False),'E2':('pad3','ra',False),
           'E2log':('pad3','log',False),'E3':('pad3','band',False),
           'E4':('pad3','ra',True),'E5':('pad3','band',True)}
    aug,consistency,conditional=modes[a.stage]
    opts=original.copy()
    opts.update(seed=a.seed,resolution_aug=aug,consistency=consistency,conditional_cdc=conditional,
                lambda_cs=1.0,lambda_band=0.2,consistency_start_epoch=5,consistency_ramp_epochs=5,
                model_name=a.name or (f'{a.stage}_pretrained_s{a.seed}'+('_smoke' if a.smoke else '')),
                load_weights_folder=None)
    if a.smoke:opts['num_epochs']=1
    if a.lambda_cs is not None:opts['lambda_cs']=a.lambda_cs
    if a.lambda_band is not None:opts['lambda_band']=a.lambda_band
    parser=LiteMonoOptions().parser
    command=[sys.executable,'train.py']
    for action in parser._actions:
        if action.dest in ['help','load_weights_folder','models_to_load'] or action.dest not in opts:continue
        value=opts[action.dest]
        if value is None:continue
        flag=next((s for s in action.option_strings if s.startswith('--')),None)
        if flag is None:continue
        if isinstance(action,argparse._StoreTrueAction):
            if value:command.append(flag)
        elif isinstance(action,argparse._StoreFalseAction):
            if not value:command.append(flag)
        elif isinstance(value,list):command += [flag]+[str(x) for x in value]
        else:command += [flag,str(value)]
    parser.parse_args(command[2:]) # catch unsupported flags before starting a job
    print(shlex.join(command),flush=True)
    if not a.run:
        print('Dry run only. Review the command, then add --run.');return
    run_dir=Path(opts['log_dir']).expanduser()/opts['model_name']
    if run_dir.exists():raise FileExistsError(f'Use a new model_name: {run_dir}')
    run_dir.mkdir(parents=True)
    weight_hash=hashlib.sha256()
    with Path(opts['mypretrain']).expanduser().open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):weight_hash.update(block)
    record=dict(pretrain_sha256=weight_hash.hexdigest(),reference_opt=str(path.resolve()),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                command=command,stage=a.stage,smoke=a.smoke)
    (run_dir/'br_launch.json').write_text(json.dumps(record,indent=2))
    with (run_dir/'console.log').open('w',buffering=1) as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line)
            code=process.wait()
        except KeyboardInterrupt:
            process.terminate();process.wait();raise
    if code:raise SystemExit(code)

if __name__=='__main__':main()
```

## 附录 E：本次实际验证范围

1. 逐文件替换、Python 语法、命令参数与文档安装器重放。
2. CPU 合成图像的 L/M/H 数据流程；none/ra/log/band 四种 loss 的普通/条件网络前向与反向；repeat3 控制。
3. 解析平面、尺度偏移、mask、条件卷积参考退化与 batch 一致性、共享基座初始化。
4. 普通/条件模型的真实 Trainer 构造、单步 Depth/Pose 优化、checkpoint 保存、strict 重载、两输入尺寸推理、合成 GT 指标和缓存复用。
5. Pose 模式切换、直接参数计数。

这些检查支持代码连接与基本数学行为，没有证明真实 KITTI 精度、跨域泛化或 GPU 实时性。CPU 验证使用独立的 torch 2.5.1+cpu / torchvision 0.20.1+cpu 环境，与你的 GPU 环境不同，不需要你为了本手册重装已有环境。
