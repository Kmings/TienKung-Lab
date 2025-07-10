## Regularization Reward
1. track_lin_vel_xy_yaw_frame_exp(env: BaseEnv | TienKungEnv,std: float,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    数学表达式说明：
    1. 首先将世界坐标系下的线速度 $\mathbf{v}_{world}$ 转换到偏航坐标系下，得到 $\mathbf{v}_{yaw}$。
    2. 计算期望线速度 $\mathbf{v}_{cmd}$ 与偏航坐标系下线速度 $\mathbf{v}_{yaw}$ 在 xy 方向的误差平方和 $E$：
       $E = \sum_{i=0}^{1} (v_{cmd, i} - v_{yaw, i})^2$
    3. 根据误差 $E$ 计算指数奖励 $R$：
       $R = \exp\left(-\frac{E}{\sigma^2}\right)$
       其中 $\sigma$ 是传入的标准差 `std`。
    该奖励函数鼓励机器人在 xy 方向上跟踪期望的线速度，同时通过指数衰减控制奖励的大小。 
    当期望线速度与机器人实际线速度接近时，指数奖励接近 1；当期望线速度与机器人实际线速度差异较大时，指数奖励接近 0。
    这个函数的设计考虑了机器人需要在 xy 平面内准确控制线速度，同时考虑到速度差异对奖励的影响。

2. track_ang_vel_z_world_exp(env: BaseEnv | TienKungEnv,std: float,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    数学表达式说明：
    1. 计算期望角速度 $\omega_{cmd}$ 与世界坐标系下 z 轴角速度 $\omega_{z, world}$ 的误差平方 $E$：
       $E = (\omega_{cmd} - \omega_{z, world})^2$
    2. 根据误差 $E$ 计算指数奖励 $R$：
       $R = \exp\left(-\frac{E}{\sigma^2}\right)$
       其中 $\sigma$ 是传入的标准差 `std`。
    该奖励函数鼓励机器人在世界坐标系下跟踪期望的 z 轴角速度，误差越小奖励越接近 1，误差越大奖励越接近 0。

3. lin_vel_z_l2(env: BaseEnv | TienKungEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    数学表达式说明：
    设机器人在本体坐标系下的 z 轴线速度为 $v_{z, body}$，则计算结果 $R$ 为：
    $R = v_{z, body}^2$
    该函数用于计算机器人在本体坐标系下 z 轴线速度的平方值，可用于衡量机器人在 z 轴方向的线速度状态。

4. ang_vel_xy_l2(env: BaseEnv | TienKungEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    数学表达式说明：
    设机器人在本体坐标系下 x 轴角速度为 $\omega_{x, body}$，y 轴角速度为 $\omega_{y, body}$，则计算结果 $R$ 为：
    $R = \omega_{x, body}^2 + \omega_{y, body}^2$
    该奖励函数鼓励机器人在 xy 平面内保持角速度为 0，即机器人不旋转。
    该函数用于计算机器人在本体坐标系下 xy 轴角速度平方和，可用于评估机器人在 xy 轴方向的旋转状态。

5. energy(env: BaseEnv | TienKungEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    数学表达式说明：
    设关节施加扭矩向量为 $\mathbf{\tau}$，关节速度向量为 $\mathbf{v}$，则计算结果 $R$ 为：
    $R = \left\lVert \mathbf{\tau} \odot \mathbf{v} \right\rVert$
    其中 $\odot$ 表示逐元素相乘，$\left\lVert \cdot \right\rVert$ 表示向量的范数。
    该函数用于计算关节施加扭矩与关节速度乘积的范数，可反映机器人关节运动时消耗的能量情况。

6. joint_acc_l2(env: BaseEnv | TienKungEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    数学表达式说明：
    设指定关节的加速度向量为 $\mathbf{a}$，则计算结果 $R$ 为：
    $R = \sum_{i} a_{i}^2$
    其中 $i$ 表示指定关节的索引。
    该函数用于计算机器人指定关节加速度的平方和，可用于评估机器人关节运动的加速度情况。

7. action_rate_l2(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    数学表达式说明：
    设当前动作向量为 $\mathbf{a}_{t}$，上一次动作向量为 $\mathbf{a}_{t - 1}$，则计算结果 $R$ 为：
    $R = \sum_{i} (a_{t, i} - a_{t - 1, i})^2$
    其中 $i$ 表示动作向量的索引。
    该函数用于计算相邻两次动作之间差值的平方和，可用于衡量机器人动作的变化幅度。

8. undesired_contacts(env: BaseEnv | TienKungEnv,threshold: float,sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor")) -> torch.Tensor:
    数学表达式说明：
    设接触传感器的历史接触力数据为 $\mathbf{F}_{history}$，对于每个环境，计算：
    1. 先计算 $\mathbf{F}_{norm} = \max_{t} \left\lVert \mathbf{F}_{history, t, body\_ids} \right\rVert$，其中 $t$ 表示时间步，$\left\lVert \cdot \right\rVert$ 表示向量的范数。
    2. 再判断 $\mathbf{F}_{norm} > threshold$，得到布尔值 $\mathbf{b}$。
    3. 最后计算 $R = \sum_{i} b_{i}$，其中 $i$ 表示环境的索引。
    该函数用于判断是否存在超过阈值的非期望接触，可帮助检测机器人是否发生了不期望的碰撞等情况。

9. is_flying(env: BaseEnv | TienKungEnv,threshold: float,sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor")) -> torch.Tensor:
    数学表达式说明：
    设接触传感器的历史接触力数据为 $\mathbf{F}_{history}$，对于每个环境，计算：
    1. 先计算 $\mathbf{F}_{norm} = \max_{t} \left\lVert \mathbf{F}_{history, t, body\_ids} \right\rVert$，其中 $t$ 表示时间步，$\left\lVert \cdot \right\rVert$ 表示向量的范数。
    2. 再判断 $\mathbf{F}_{norm} > threshold$，得到布尔值 $\mathbf{b}$。
    3. 计算 $\sum_{i} b_{i}$，其中 $i$ 表示环境的索引。
    4. 最后判断 $\sum_{i} b_{i} < 0.5$，得到最终结果。
    该函数用于判断机器人是否处于飞行状态，可通过接触力情况来间接判断机器人是否离开地面。

10. flat_orientation_l2(env: BaseEnv | TienKungEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    数学表达式说明：
    设机器人在本体坐标系下投影重力向量为 $\mathbf{g}_{proj}$，其 xy 分量为 $g_{proj, x}$ 和 $g_{proj, y}$，则计算结果 $R$ 为：
    $R = g_{proj, x}^2 + g_{proj, y}^2$
    该函数用于计算机器人在本体坐标系下投影重力向量 xy 分量的平方和，可用于评估机器人的姿态是否平稳。

11. is_terminated(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    数学表达式说明：
    设重置标志为 $reset\_buf$，超时标志为 $time\_out\_buf$，则计算结果 $R$ 为：
    $R = reset\_buf \land \neg time\_out\_buf$
    其中 $\land$ 表示逻辑与，$\neg$ 表示逻辑非。
    该函数用于对非因情节超时导致终止的情节进行惩罚，可帮助区分不同的情节终止原因。

12. feet_air_time_positive_biped(env: BaseEnv | TienKungEnv,threshold: float,sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    数学表达式说明：
    1. 从接触传感器获取双脚当前的腾空时间 $\mathbf{t}_{air}$ 和接触时间 $\mathbf{t}_{contact}$。
    2. 判断双脚是否处于接触状态，得到布尔向量 $\mathbf{c}$，$c_i = t_{contact, i} > 0.0$。
    3. 根据接触状态选择接触时间或腾空时间，得到 $\mathbf{t}_{mode}$，$t_{mode, i} = \begin{cases}t_{contact, i}, & c_i \\ t_{air, i}, & \neg c_i\end{cases}$。
    4. 判断是否处于单脚支撑状态，得到布尔向量 $\mathbf{s}$，$s_j = \sum_{i} c_{i, j} == 1$。
    5. 计算单脚支撑时的奖励 $\mathbf{r}$，$r_j = \min_{i} \begin{cases}t_{mode, i, j}, & s_j \\ 0.0, & \neg s_j\end{cases}$。
    6. 对奖励进行裁剪，$R = \min(r, threshold)$。
    该函数用于计算双足机器人单脚支撑时的奖励，奖励与单脚支撑时间相关，鼓励机器人在运动过程中保持单脚支撑的姿态。

13. feet_slide(env: BaseEnv | TienKungEnv,sensor_cfg: SceneEntityCfg,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    数学表达式说明：
    1. 从接触传感器获取接触力数据 $\mathbf{F}_{contact}$，计算是否有接触力大于 1.0 的情况，得到布尔向量 $\mathbf{c}$，$c_j = \max_{t} \left\lVert \mathbf{F}_{contact, t, body\_ids, j} \right\rVert > 1.0$。
    2. 获取指定身体部位在世界坐标系下的线速度 $\mathbf{v}_{body}$。
    3. 计算脚滑动的奖励 $R$，$R = \sum_{i} \left\lVert \mathbf{v}_{body, i} \right\rVert \cdot c_i$。
    该函数用于计算机器人脚滑动的奖励，可通过接触力和线速度情况来评估机器人脚的滑动状态。

14. body_force(env: BaseEnv | TienKungEnv,sensor_cfg: SceneEntityCfg,threshold: float = 500,max_reward: float = 400) -> torch.Tensor:
    数学表达式说明：
    1. 从接触传感器获取接触力数据 $\mathbf{F}_{contact}$，提取法向接触力的范数 $\mathbf{F}_{norm}$。
    2. 对于 $\mathbf{F}_{norm}$ 中小于阈值 $threshold$ 的元素，将其置为 0，即 $F_{norm, i} = \begin{cases}F_{norm, i} - threshold, & F_{norm, i} > threshold \\ 0, & F_{norm, i} \leq threshold\end{cases}$。
    3. 对处理后的 $\mathbf{F}_{norm}$ 进行裁剪，$R = \min(\max(F_{norm, i}, 0), max\_reward)$。
    该函数用于计算机器人身体受到的接触力奖励，可对超过阈值的接触力进行奖励，同时限制最大奖励值。

15. joint_deviation_l1(env: BaseEnv | TienKungEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    数学表达式说明：
    1. 计算关节位置与默认位置的偏差 $\mathbf{\Delta \theta} = \mathbf{\theta}_{current} - \mathbf{\theta}_{default}$，其中 $\mathbf{\theta}_{current}$ 是当前关节位置，$\mathbf{\theta}_{default}$ 是默认关节位置。
    2. 判断命令速度是否接近 0，得到布尔向量 $\mathbf{v}_{flag}$，$v_{flag, j} = \left\lVert \mathbf{v}_{cmd, j, :2} \right\rVert + |\omega_{cmd, j}| < 0.1$。
    3. 计算关节位置偏差的 L1 范数，并乘以速度标志，$R = \sum_{i} |\Delta \theta_{i}| \cdot v_{flag}$。
    该函数用于计算机器人关节位置与默认位置偏差的 L1 范数，且在命令速度较低时才进行计算，可帮助机器人在静止时保持关节位置稳定。

16. body_orientation_l2(env: BaseEnv | TienKungEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    数学表达式说明：
    1. 将重力向量 $\mathbf{g}_{world}$ 转换到指定身体部位的坐标系下，得到 $\mathbf{g}_{body}$，$\mathbf{g}_{body} = \text{quat\_rotate\_inverse}(\mathbf{q}_{body}, \mathbf{g}_{world})$，其中 $\mathbf{q}_{body}$ 是指定身体部位的四元数。
    2. 计算身体朝向与重力向量在 xy 平面偏差的 L2 范数 $R$，$R = \sum_{i=0}^{1} g_{body, i}^2$。
    该函数用于计算机器人身体朝向与重力向量在 xy 平面偏差的 L2 范数，可评估机器人的姿态是否符合预期。

17. feet_stumble(env: BaseEnv | TienKungEnv,sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    数学表达式说明：
    1. 从接触传感器获取接触力数据 $\mathbf{F}_{contact}$，计算脚的水平接触力范数 $\mathbf{F}_{horizontal}$，$F_{horizontal, i} = \left\lVert \mathbf{F}_{contact, i, :2} \right\rVert$。
    2. 提取脚的垂直接触力 $\mathbf{F}_{vertical}$，$F_{vertical, i} = |F_{contact, i, 2}|$。
    3. 判断是否有脚的水平接触力大于垂直接触力 5 倍的情况，得到布尔向量 $\mathbf{b}$，$b_j = \exists_{i} F_{horizontal, i, j} > 5 \cdot F_{vertical, i, j}$。
    该函数用于判断机器人脚是否有绊倒的情况，可通过接触力的水平和垂直分量关系来检测。

18. feet_too_near_humanoid(env: BaseEnv | TienKungEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),threshold: float = 0.2) -> torch.Tensor:
    数学表达式说明：
    1. 获取双脚在世界坐标系下的位置 $\mathbf{p}_{left}$ 和 $\mathbf{p}_{right}$。
    2. 计算双脚之间的距离 $d = \left\lVert \mathbf{p}_{left} - \mathbf{p}_{right} \right\rVert$。
    3. 计算双脚距离过近的惩罚值 $R = \max(threshold - d, 0)$。
    该函数用于计算人形机器人双脚距离过近的惩罚值，可帮助机器人保持双脚适当的间距。

19. ankle_torque(env: TienKungEnv) -> torch.Tensor:
    数学表达式说明：
    1. 从环境中获取机器人踝关节施加的扭矩 $\mathbf{\tau}_{ankle}$，索引为 `env.ankle_joint_ids`。
    2. 计算踝关节施加扭矩的平方和 $R = \sum_{i} \tau_{ankle, i}^2$。
    该函数用于对脚踝关节施加的大扭矩进行惩罚，鼓励机器人减少不必要的脚踝关节扭矩。

20. ankle_action(env: TienKungEnv) -> torch.Tensor:
    数学表达式说明：
    1. 首先获取环境中机器人动作的张量 `env.action`，其维度可能包含批量维度和关节动作维度。
    2. 从 `env.action` 中提取出踝关节对应的动作值，索引为 `env.ankle_joint_ids`。
    3. 对提取出的踝关节动作值取绝对值，即 $|\mathbf{a}_{ankle}|$，其中 $\mathbf{a}_{ankle}$ 是踝关节动作向量。
    4. 沿着指定维度（`dim=1`）对绝对值后的动作值进行求和，得到每个样本的总惩罚值 $P$：
       $P = \sum_{i} |a_{ankle, i}|$
    该函数用于惩罚踝关节的动作，鼓励机器人减少不必要的踝关节动作。惩罚值越大，说明踝关节动作越剧烈。
    这个函数的设计考虑了机器人在运动过程中需要尽量保持踝关节动作的平稳，以提高运动的稳定性和效率。

21. hip_roll_action(env: TienKungEnv) -> torch.Tensor:
    数学表达式说明：
    1. 从环境中获取机器人动作的张量 `env.action`。
    2. 提取髋关节滚动关节对应的动作值，索引为 `[env.left_leg_ids[0], env.right_leg_ids[0]]`。
    3. 对提取出的动作值取绝对值，得到 $|\mathbf{a}_{hip\_roll}|$。
    4. 沿着指定维度（`dim=1`）对绝对值后的动作值进行求和，得到 $R = \sum_{i} |a_{hip\_roll, i}|$。
    该函数用于对髋关节滚动关节的动作进行惩罚，促使机器人减少不必要的髋关节滚动动作。

22. hip_yaw_action(env: TienKungEnv) -> torch.Tensor:
    数学表达式说明：
    1. 从环境中获取机器人动作的张量 `env.action`。
    2. 提取髋关节偏航关节对应的动作值，索引为 `[env.left_leg_ids[2], env.right_leg_ids[2]]`。
    3. 对提取出的动作值取绝对值，得到 $|\mathbf{a}_{hip\_yaw}|$。
    4. 沿着指定维度（`dim=1`）对绝对值后的动作值进行求和，得到 $R = \sum_{i} |a_{hip\_yaw, i}|$。
    该函数用于对髋关节偏航关节的动作进行惩罚，鼓励机器人减少不必要的髋关节偏航动作。

23. feet_y_distance(env: TienKungEnv) -> torch.Tensor:
    数学表达式说明：
    1. 计算左脚相对于根节点的位置 $\mathbf{p}_{left\_rel} = \mathbf{p}_{left} - \mathbf{p}_{root}$，其中 $\mathbf{p}_{left}$ 是左脚位置，$\mathbf{p}_{root}$ 是根节点位置。
    2. 计算右脚相对于根节点的位置 $\mathbf{p}_{right\_rel} = \mathbf{p}_{right} - \mathbf{p}_{root}$。
    3. 将左脚位置转换到根节点坐标系下，$\mathbf{p}_{left\_b} = \text{quat\_apply}(\text{quat\_conjugate}(\mathbf{q}_{root}), \mathbf{p}_{left\_rel})$，其中 $\mathbf{q}_{root}$ 是根节点四元数。
    4. 将右脚位置转换到根节点坐标系下，$\mathbf{p}_{right\_b} = \text{quat\_apply}(\text{quat\_conjugate}(\mathbf{q}_{root}), \mathbf{p}_{right\_rel})$。
    5. 计算双脚在根节点坐标系下 y 轴的距离与目标距离的差值的绝对值 $d_y = |p_{left\_b, y} - p_{right\_b, y} - 0.299|$。
    6. 判断命令的 y 轴速度是否较低，得到布尔向量 $\mathbf{v}_{flag}$，$v_{flag, j} = |v_{cmd, j, 1}| < 0.1$。
    7. 计算双脚在 y 轴上距离的惩罚值 $R = d_y \cdot \mathbf{v}_{flag}$。
    该函数在命令的 y 轴速度较低时，对双脚在 y 轴上的距离进行惩罚，以保持合理间距。

## Periodic gait-based reward function
24. gait_clock(phase: torch.Tensor,air_ratio: torch.Tensor,delta_t: float) -> tuple[torch.Tensor, torch.Tensor]:
    数学表达式说明：
    1. 定义摆动相标志 $\mathbf{s}_{swing}$：
       $\mathbf{s}_{swing} = (\mathbf{phase} \geq \delta_t) \land (\mathbf{phase} \leq (\mathbf{air\_ratio} - \delta_t))$
    2. 定义支撑相标志 $\mathbf{s}_{stand}$：
       $\mathbf{s}_{stand} = (\mathbf{phase} \geq (\mathbf{air\_ratio} + \delta_t)) \land (\mathbf{phase} \leq (1 - \delta_t))$
    3. 定义过渡阶段标志 $\mathbf{s}_{trans1}$、$\mathbf{s}_{trans2}$、$\mathbf{s}_{trans3}$：
       $\mathbf{s}_{trans1} = \mathbf{phase} < \delta_t$
       $\mathbf{s}_{trans2} = (\mathbf{phase} > (\mathbf{air\_ratio} - \delta_t)) \land (\mathbf{phase} < (\mathbf{air\_ratio} + \delta_t))$
       $\mathbf{s}_{trans3} = \mathbf{phase} > (1 - \delta_t)$
    4. 计算摆动相时钟信号 $\mathbf{I}_{frc}$：
       $\mathbf{I}_{frc} = 1.0 \cdot \mathbf{s}_{swing} + (0.5 + \frac{\mathbf{phase}}{2 \delta_t}) \cdot \mathbf{s}_{trans1} - \frac{\mathbf{phase} - \mathbf{air\_ratio} - \delta_t}{2.0 \delta_t} \cdot \mathbf{s}_{trans2} + 0.0 \cdot \mathbf{s}_{stand} + \frac{\mathbf{phase} - 1 + \delta_t}{2 \delta_t} \cdot \mathbf{s}_{trans3}$
    5. 计算支撑相时钟信号 $\mathbf{I}_{spd}$：
       $\mathbf{I}_{spd} = 1.0 - \mathbf{I}_{frc}$
    该函数生成用于脚摆动相和支撑相的周期性步态时钟信号，在摆动相和支撑相之间进行平滑过渡。

25. gait_feet_frc_perio(env: TienKungEnv,delta_t: float = 0.02) -> torch.Tensor:
    数学表达式说明：
    1. 获取左脚摆动相的掩码 $\mathbf{m}_{left\_frc}$，$\mathbf{m}_{left\_frc} = \text{gait\_clock}(\mathbf{phase}_{left}, \mathbf{air\_ratio}_{left}, \delta_t)[0]$。
    2. 获取右脚摆动相的掩码 $\mathbf{m}_{right\_frc}$，$\mathbf{m}_{right\_frc} = \text{gait\_clock}(\mathbf{phase}_{right}, \mathbf{air\_ratio}_{right}, \delta_t)[0]$。
    3. 计算左脚在摆动相的脚力得分 $\mathbf{r}_{left\_frc}$，$\mathbf{r}_{left\_frc} = \mathbf{m}_{left\_frc} \cdot \exp(-200 \cdot \mathbf{F}_{left}^2)$，其中 $\mathbf{F}_{left}$ 是左脚平均脚力。
    4. 计算右脚在摆动相的脚力得分 $\mathbf{r}_{right\_frc}$，$\mathbf{r}_{right\_frc} = \mathbf{m}_{right\_frc} \cdot \exp(-200 \cdot \mathbf{F}_{right}^2)$，其中 $\mathbf{F}_{right}$ 是右脚平均脚力。
    5. 计算最终奖励 $R = \mathbf{r}_{left\_frc} + \mathbf{r}_{right\_frc}$。
    该函数在步态的摆动相期间惩罚脚受到的力，脚力越大奖励越小。

26. gait_feet_spd_perio(env: TienKungEnv,delta_t: float = 0.02) -> torch.Tensor:
    数学表达式说明：
    1. 获取左脚支撑相的掩码 $\mathbf{m}_{left\_spd}$，$\mathbf{m}_{left\_spd} = \text{gait\_clock}(\mathbf{phase}_{left}, \mathbf{air\_ratio}_{left}, \delta_t)[1]$。
    2. 获取右脚支撑相的掩码 $\mathbf{m}_{right\_spd}$，$\mathbf{m}_{right\_spd} = \text{gait\_clock}(\mathbf{phase}_{right}, \mathbf{air\_ratio}_{right}, \delta_t)[1]$。
    3. 计算左脚在支撑相的脚速得分 $\mathbf{r}_{left\_spd}$，$\mathbf{r}_{left\_spd} = \mathbf{m}_{left\_spd} \cdot \exp(-100 \cdot \mathbf{v}_{left}^2)$，其中 $\mathbf{v}_{left}$ 是左脚平均脚速。
    4. 计算右脚在支撑相的脚速得分 $\mathbf{r}_{right\_spd}$，$\mathbf{r}_{right\_spd} = \mathbf{m}_{right\_spd} \cdot \exp(-100 \cdot \mathbf{v}_{right}^2)$，其中 $\mathbf{v}_{right}$ 是右脚平均脚速。
    5. 计算最终奖励 $R = \mathbf{r}_{left\_spd} + \mathbf{r}_{right\_spd}$。
    该函数在步态的支撑相期间惩罚脚的速度，脚速越大奖励越小。

27. gait_feet_frc_support_perio(env: TienKungEnv,delta_t: float = 0.02) -> torch.Tensor:
    数学表达式说明：
    1. 获取左脚支撑相的掩码 $\mathbf{m}_{left\_frc\_support}$，$\mathbf{m}_{left\_frc\_support} = \text{gait\_clock}(\mathbf{phase}_{left}, \mathbf{air\_ratio}_{left}, \delta_t)[1]$。
    2. 获取右脚支撑相的掩码 $\mathbf{m}_{right\_frc\_support}$，$\mathbf{m}_{right\_frc\_support} = \text{gait\_clock}(\mathbf{phase}_{right}, \mathbf{air\_ratio}_{right}, \delta_t)[1]$。
    3. 计算左脚在支撑相的脚支撑力得分 $\mathbf{r}_{left\_frc\_support}$，$\mathbf{r}_{left\_frc\_support} = \mathbf{m}_{left\_frc\_support} \cdot (1 - \exp(-10 \cdot \mathbf{F}_{left}^2))$，其中 $\mathbf{F}_{left}$ 是左脚平均脚力。
    4. 计算右脚在支撑相的脚支撑力得分 $\mathbf{r}_{right\_frc\_support}$，$\mathbf{r}_{right\_frc\_support} = \mathbf{m}_{right\_frc\_support} \cdot (1 - \exp(-10 \cdot \mathbf{F}_{right}^2))$，其中 $\mathbf{F}_{right}$ 是右脚平均脚力。
    5. 计算最终奖励 $R = \mathbf{r}_{left\_frc\_support} + \mathbf{r}_{right\_frc\_support}$。
    该函数奖励在支撑相期间提供适当支撑力的行为，脚支撑力越合适奖励越大。
