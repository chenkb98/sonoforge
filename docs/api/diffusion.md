# diffusion


## `linear_schedule`

构造线性 beta 噪声方差调度，所有值严格位于 (0, 1)，并限制最大步数。单步时使用起始方差。

## `cosine_schedule`

离散化余弦累计信号比例，将最后一步 beta 截断到 0.999，避免完全为零的累计 alpha 引起除零。

## `validate_schedule`

校验 beta 向量类型、有限性和范围，并拒绝累计 alpha 小到无法稳定反演的调度。返回副本以隔离调用者修改。
