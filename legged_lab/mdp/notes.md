## Regularization Reward
1. <span style="font-size: 1.2em; font-weight: bold;">`track_lin_vel_xy_yaw_frame_exp`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 首先将世界坐标系下的线速度 <span style="color: #007BFF;">$\mathbf{v}_{world}$</span> 转换到偏航坐标系下，得到 <span style="color: #007BFF;">$\mathbf{v}_{yaw}$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 计算期望线速度 <span style="color: #007BFF;">$\mathbf{v}_{cmd}$</span> 与偏航坐标系下线速度 <span style="color: #007BFF;">$\mathbf{v}_{yaw}$</span> 在 xy 方向的误差平方和 <span style="color: #007BFF;">$E$</span>：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$E = \sum_{i=0}^{1} (v_{cmd, i} - v_{yaw, i})^2$</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 根据误差 <span style="color: #007BFF;">$E$</span> 计算指数奖励 <span style="color: #007BFF;">$R$</span>：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$R = \exp\left(-\frac{E}{\sigma^2}\right)$，其中 <span style="color: #007BFF;">$\sigma$</span> 是传入的标准差 `std`。</span>  
    该奖励函数鼓励机器人在 xy 方向上跟踪期望的线速度，同时通过指数衰减控制奖励的大小。 
    当期望线速度与机器人实际线速度接近时，指数奖励接近 1；当期望线速度与机器人实际线速度差异较大时，指数奖励接近 0。

2. <span style="font-size: 1.2em; font-weight: bold;">`track_ang_vel_z_world_exp`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 计算期望角速度 <span style="color: #007BFF;">$\omega_{cmd}$</span> 与世界坐标系下 z 轴角速度 <span style="color: #007BFF;">$\omega_{z, world}$</span> 的误差平方 <span style="color: #007BFF;">$E$</span>：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$E = (\omega_{cmd} - \omega_{z, world})^2$</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 根据误差 <span style="color: #007BFF;">$E$</span> 计算指数奖励 <span style="color: #007BFF;">$R$</span>：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$R = \exp\left(-\frac{E}{\sigma^2}\right)$，其中 <span style="color: #007BFF;">$\sigma$</span> 是传入的标准差 `std`。</span>  
    该奖励函数鼓励机器人在世界坐标系下跟踪期望的 z 轴角速度，误差越小奖励越接近 1，误差越大奖励越接近 0。

3. <span style="font-size: 1.2em; font-weight: bold;">`lin_vel_z_l2`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 设机器人在本体坐标系下的 z 轴线速度为 <span style="color: #007BFF;">$v_{z, body}$</span>，则计算结果 <span style="color: #007BFF;">$R$</span> 为：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$R = v_{z, body}^2$</span>  
    该函数用于计算机器人在本体坐标系下 z 轴线速度的平方值，可用于衡量机器人在 z 轴方向的线速度状态。

4. <span style="font-size: 1.2em; font-weight: bold;">`ang_vel_xy_l2`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 设机器人在本体坐标系下 x 轴角速度为 <span style="color: #007BFF;">$\omega_{x, body}$</span>，y 轴角速度为 <span style="color: #007BFF;">$\omega_{y, body}$</span>，则计算结果 <span style="color: #007BFF;">$R$</span> 为：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$R = \omega_{x, body}^2 + \omega_{y, body}^2$</span>  
    该奖励函数鼓励机器人在 xy 平面内保持角速度为 0，即机器人不旋转。
    该函数用于计算机器人在本体坐标系下 xy 轴角速度平方和，可用于评估机器人在 xy 轴方向的旋转状态。

5. <span style="font-size: 1.2em; font-weight: bold;">`energy`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 设关节施加扭矩向量为 <span style="color: #007BFF;">$\mathbf{\tau}$</span>，关节速度向量为 <span style="color: #007BFF;">$\mathbf{v}$</span>，则计算结果 <span style="color: #007BFF;">$R$</span> 为：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$R = \left\lVert \mathbf{\tau} \odot \mathbf{v} \right\rVert$，其中 <span style="color: #007BFF;">$\odot$</span> 表示逐元素相乘，<span style="color: #007BFF;">$\left\lVert \cdot \right\rVert$</span> 表示向量的范数。</span>  
    该函数用于计算关节施加扭矩与关节速度乘积的范数，可反映机器人关节运动时消耗的能量情况。

6. <span style="font-size: 1.2em; font-weight: bold;">`joint_acc_l2`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 设指定关节的加速度向量为 <span style="color: #007BFF;">$\mathbf{a}$</span>，则计算结果 <span style="color: #007BFF;">$R$</span> 为：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$R = \sum_{i} a_{i}^2$</span>  
    该函数用于计算机器人指定关节加速度的平方和，可用于评估机器人关节运动的加速度情况。

7. <span style="font-size: 1.2em; font-weight: bold;">`action_rate_l2`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 设当前动作向量为 <span style="color: #007BFF;">$\mathbf{a}_{t}$</span>，上一次动作向量为 <span style="color: #007BFF;">$\mathbf{a}_{t - 1}$</span>，则计算结果 <span style="color: #007BFF;">$R$</span> 为：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$R = \sum_{i} (a_{t, i} - a_{t - 1, i})^2$</span>  
    该函数用于计算相邻两次动作之间差值的平方和，可用于衡量机器人动作的变化幅度。

8. <span style="font-size: 1.2em; font-weight: bold;">`undesired_contacts`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 设接触传感器的历史接触力数据为 <span style="color: #007BFF;">$\mathbf{F}_{history}$</span>，对于每个环境，计算：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 先计算 <span style="color: #007BFF;">$\mathbf{F}_{norm} = \max_{t} \left\lVert \mathbf{F}_{history, t, body\_ids} \right\rVert$</span>，其中 <span style="color: #007BFF;">$t$</span> 表示时间步，<span style="color: #007BFF;">$\left\lVert \cdot \right\rVert$</span> 表示向量的范数。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 再判断 <span style="color: #007BFF;">$\mathbf{F}_{norm} > threshold$</span>，得到布尔值 <span style="color: #007BFF;">$\mathbf{b}$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iv. 最后计算 <span style="color: #007BFF;">$R = \sum_{i} b_{i}$</span>，其中 <span style="color: #007BFF;">$i$</span> 表示环境的索引。</span>  
    该函数用于判断是否存在超过阈值的非期望接触，可帮助检测机器人是否发生了不期望的碰撞等情况。

9. <span style="font-size: 1.2em; font-weight: bold;">`is_flying`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 设接触传感器的历史接触力数据为 <span style="color: #007BFF;">$\mathbf{F}_{history}$</span>，对于每个环境，计算：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 先计算 <span style="color: #007BFF;">$\mathbf{F}_{norm} = \max_{t} \left\lVert \mathbf{F}_{history, t, body\_ids} \right\rVert$</span>，其中 <span style="color: #007BFF;">$t$</span> 表示时间步，<span style="color: #007BFF;">$\left\lVert \cdot \right\rVert$</span> 表示向量的范数。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 再判断 <span style="color: #007BFF;">$\mathbf{F}_{norm} > threshold$</span>，得到布尔值 <span style="color: #007BFF;">$\mathbf{b}$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iv. 计算 <span style="color: #007BFF;">$\sum_{i} b_{i}$</span>，其中 <span style="color: #007BFF;">$i$</span> 表示环境的索引。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">v. 最后判断 <span style="color: #007BFF;">$\sum_{i} b_{i} < 0.5$</span>，得到最终结果。</span>  
    该函数用于判断机器人是否处于飞行状态，可通过接触力情况来间接判断机器人是否离开地面。

10. <span style="font-size: 1.2em; font-weight: bold;">`flat_orientation_l2`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 设机器人在本体坐标系下投影重力向量为 <span style="color: #007BFF;">$\mathbf{g}_{proj}$</span>，其 xy 分量为 <span style="color: #007BFF;">$g_{proj, x}$</span> 和 <span style="color: #007BFF;">$g_{proj, y}$</span>，则计算结果 <span style="color: #007BFF;">$R$</span> 为：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$R = g_{proj, x}^2 + g_{proj, y}^2$</span>  
    该函数用于计算机器人在本体坐标系下投影重力向量 xy 分量的平方和，可用于评估机器人的姿态是否平稳。

11. <span style="font-size: 1.2em; font-weight: bold;">`is_terminated`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 设重置标志为 <span style="color: #007BFF;">$reset\_buf$</span>，超时标志为 <span style="color: #007BFF;">$time\_out\_buf$</span>，则计算结果 <span style="color: #007BFF;">$R$</span> 为：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$R = reset\_buf \land \neg time\_out\_buf$</span> 其中 <span style="color: #007BFF;">$\land$</span> 表示逻辑与，<span style="color: #007BFF;">$\neg$</span> 表示逻辑非。</span>  
    该函数用于对非因情节超时导致终止的情节进行惩罚，可帮助区分不同的情节终止原因。

12. <span style="font-size: 1.2em; font-weight: bold;">`feet_air_time_positive_biped`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 从接触传感器获取双脚当前的腾空时间 <span style="color: #007BFF;">$\mathbf{t}_{air}$</span> 和接触时间 <span style="color: #007BFF;">$\mathbf{t}_{contact}$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 判断双脚是否处于接触状态，得到布尔向量 <span style="color: #007BFF;">$\mathbf{c}$</span>，<span style="color: #007BFF;">$c_i = t_{contact, i} > 0.0$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 根据接触状态选择接触时间或腾空时间，得到 <span style="color: #007BFF;">$\mathbf{t}_{mode}$</span>，<span style="color: #007BFF;">$t_{mode, i} = \begin{cases}t_{contact, i}, & c_i \\ t_{air, i}, & \neg c_i\end{cases}$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iv. 判断是否处于单脚支撑状态，得到布尔向量 <span style="color: #007BFF;">$\mathbf{s}$</span>，<span style="color: #007BFF;">$s_j = \sum_{i} c_{i, j} == 1$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">v. 计算单脚支撑时的奖励 <span style="color: #007BFF;">$\mathbf{r}$</span>，<span style="color: #007BFF;">$r_j = \min_{i} \begin{cases}t_{mode, i, j}, & s_j \\ 0.0, & \neg s_j\end{cases}$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">vi. 对奖励进行裁剪，<span style="color: #007BFF;">$R = \min(r, threshold)$</span>。</span>  
    该函数用于计算双足机器人单脚支撑时的奖励，奖励与单脚支撑时间相关，鼓励机器人在运动过程中保持单脚支撑的姿态。

13. <span style="font-size: 1.2em; font-weight: bold;">`feet_slide`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 从接触传感器获取接触力数据 <span style="color: #007BFF;">$\mathbf{F}_{contact}$</span>，计算是否有接触力大于 1.0 的情况，得到布尔向量 <span style="color: #007BFF;">$\mathbf{c}$</span>，<span style="color: #007BFF;">$c_j = \max_{t} \left\lVert \mathbf{F}_{contact, t, body\_ids, j} \right\rVert > 1.0$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 获取指定身体部位在世界坐标系下的线速度 <span style="color: #007BFF;">$\mathbf{v}_{body}$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 计算脚滑动的奖励 <span style="color: #007BFF;">$R = \sum_{i} \left\lVert \mathbf{v}_{body, i} \right\rVert \cdot c_i$</span>。</span>  
    该函数用于计算机器人脚滑动的奖励，可通过接触力和线速度情况来评估机器人脚的滑动状态。

14. <span style="font-size: 1.2em; font-weight: bold;">`body_force`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 从接触传感器获取接触力数据 <span style="color: #007BFF;">$\mathbf{F}_{contact}$</span>，提取法向接触力的范数 <span style="color: #007BFF;">$\mathbf{F}_{norm}$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 对于 <span style="color: #007BFF;">$\mathbf{F}_{norm}$</span> 中小于阈值 <span style="color: #007BFF;">$threshold$</span> 的元素，将其置为 0，即 <span style="color: #007BFF;">$F_{norm, i} = \begin{cases}F_{norm, i} - threshold, & F_{norm, i} > threshold \\ 0, & F_{norm, i} \leq threshold\end{cases}$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 对处理后的 <span style="color: #007BFF;">$\mathbf{F}_{norm}$</span> 进行裁剪，<span style="color: #007BFF;">$R = \min(\max(F_{norm, i}, 0), max\_reward)$</span>。</span>  
    该函数用于计算机器人身体受到的接触力奖励，可对超过阈值的接触力进行奖励，同时限制最大奖励值。

15. <span style="font-size: 1.2em; font-weight: bold;">`joint_deviation_l1`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 计算关节位置与默认位置的偏差 <span style="color: #007BFF;">$\mathbf{\Delta \theta} = \mathbf{\theta}_{current} - \mathbf{\theta}_{default}$</span>，其中 <span style="color: #007BFF;">$\mathbf{\theta}_{current}$</span> 是当前关节位置，<span style="color: #007BFF;">$\mathbf{\theta}_{default}$</span> 是默认关节位置。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 判断命令速度是否接近 0，得到布尔向量 <span style="color: #007BFF;">$\mathbf{v}_{flag}$</span>，<span style="color: #007BFF;">$v_{flag, j} = \left\lVert \mathbf{v}_{cmd, j, :2} \right\rVert + |\omega_{cmd, j}| < 0.1$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 计算关节位置偏差的 L1 范数，并乘以速度标志，<span style="color: #007BFF;">$R = \sum_{i} |\Delta \theta_{i}| \cdot v_{flag}$</span>。</span>  
    该函数用于计算机器人关节位置与默认位置偏差的 L1 范数，且在命令速度较低时才进行计算，可帮助机器人在静止时保持关节位置稳定。

16. <span style="font-size: 1.2em; font-weight: bold;">`body_orientation_l2`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 将重力向量 <span style="color: #007BFF;">$\mathbf{g}_{world}$</span> 转换到指定身体部位的坐标系下，得到 <span style="color: #007BFF;">$\mathbf{g}_{body}$</span>，<span style="color: #007BFF;">$\mathbf{g}_{body} = \text{quat\_rotate\_inverse}(\mathbf{q}_{body}, \mathbf{g}_{world})$</span>，其中 <span style="color: #007BFF;">$\mathbf{q}_{body}$</span> 是指定身体部位的四元数。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 计算身体朝向与重力向量在 xy 平面偏差的 L2 范数 <span style="color: #007BFF;">$R$</span>，<span style="color: #007BFF;">$R = \sum_{i=0}^{1} g_{body, i}^2$</span>。</span>  
    该函数用于计算机器人身体朝向与重力向量在 xy 平面偏差的 L2 范数，可评估机器人的姿态是否符合预期。

17. <span style="font-size: 1.2em; font-weight: bold;">`feet_stumble`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 从接触传感器获取接触力数据 <span style="color: #007BFF;">$\mathbf{F}_{contact}$</span>，计算脚的水平接触力范数 <span style="color: #007BFF;">$\mathbf{F}_{horizontal}$</span>，<span style="color: #007BFF;">$F_{horizontal, i} = \left\lVert \mathbf{F}_{contact, i, :2} \right\rVert$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 提取脚的垂直接触力 <span style="color: #007BFF;">$\mathbf{F}_{vertical}$</span>，<span style="color: #007BFF;">$F_{vertical, i} = |F_{contact, i, 2}|$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 判断是否有脚的水平接触力大于垂直接触力 5 倍的情况，得到布尔向量 <span style="color: #007BFF;">$\mathbf{b}$</span>，<span style="color: #007BFF;">$b_j = \exists_{i} F_{horizontal, i, j} > 5 \cdot F_{vertical, i, j}$</span>。</span>  
    该函数用于判断机器人脚是否有绊倒的情况，可通过接触力的水平和垂直分量关系来检测。

18. <span style="font-size: 1.2em; font-weight: bold;">`feet_too_near_humanoid`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 获取双脚在世界坐标系下的位置 <span style="color: #007BFF;">$\mathbf{p}_{left}$</span> 和 <span style="color: #007BFF;">$\mathbf{p}_{right}$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 计算双脚之间的距离 <span style="color: #007BFF;">$d = \left\lVert \mathbf{p}_{left} - \mathbf{p}_{right} \right\rVert$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 计算双脚距离过近的惩罚值 <span style="color: #007BFF;">$R = \max(threshold - d, 0)$</span>。</span>  
    该函数用于计算人形机器人双脚距离过近的惩罚值，可帮助机器人保持双脚适当的间距。

19. <span style="font-size: 1.2em; font-weight: bold;">`ankle_torque`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 从环境中获取机器人踝关节施加的扭矩 <span style="color: #007BFF;">$\mathbf{\tau}_{ankle}$</span>，索引为 <span style="color: #007BFF;">`env.ankle_joint_ids`</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 计算踝关节施加扭矩的平方和 <span style="color: #007BFF;">$R = \sum_{i} \tau_{ankle, i}^2$</span>。</span>  
    该函数用于对脚踝关节施加的大扭矩进行惩罚，鼓励机器人减少不必要的脚踝关节扭矩。

20. <span style="font-size: 1.2em; font-weight: bold;">`ankle_action`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 首先获取环境中机器人动作的张量 <span style="color: #007BFF;">`env.action`</span>，其维度可能包含批量维度和关节动作维度。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 从 <span style="color: #007BFF;">`env.action`</span> 中提取出踝关节对应的动作值，索引为 <span style="color: #007BFF;">`env.ankle_joint_ids`</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 对提取出的踝关节动作值取绝对值，即 <span style="color: #007BFF;">$|\mathbf{a}_{ankle}|$</span>，其中 <span style="color: #007BFF;">$\mathbf{a}_{ankle}$</span> 是踝关节动作向量。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iv. 沿着指定维度（<span style="color: #007BFF;">`dim=1`</span>）对绝对值后的动作值进行求和，得到每个样本的总惩罚值 <span style="color: #007BFF;">$P$</span>：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$P = \sum_{i} |a_{ankle, i}|$</span>  
    该函数用于惩罚踝关节的动作，鼓励机器人减少不必要的踝关节动作。惩罚值越大，说明踝关节动作越剧烈。
    这个函数的设计考虑了机器人在运动过程中需要尽量保持踝关节动作的平稳，以提高运动的稳定性和效率。

21. <span style="font-size: 1.2em; font-weight: bold;">`hip_roll_action`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 从环境中获取机器人动作的张量 <span style="color: #007BFF;">`env.action`</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 提取髋关节滚动关节对应的动作值，索引为 <span style="color: #007BFF;">`[env.left_leg_ids[0], env.right_leg_ids[0]]`</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 对提取出的动作值取绝对值，得到 <span style="color: #007BFF;">$|\mathbf{a}_{hip\_roll}|$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iv. 沿着指定维度（<span style="color: #007BFF;">`dim=1`</span>）对绝对值后的动作值进行求和，得到 <span style="color: #007BFF;">$R = \sum_{i} |a_{hip\_roll, i}|$</span>。</span>  
    该函数用于对髋关节滚动关节的动作进行惩罚，促使机器人减少不必要的髋关节滚动动作。

22. <span style="font-size: 1.2em; font-weight: bold;">`hip_yaw_action`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 从环境中获取机器人动作的张量 <span style="color: #007BFF;">`env.action`</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 提取髋关节偏航关节对应的动作值，索引为 <span style="color: #007BFF;">`[env.left_leg_ids[2], env.right_leg_ids[2]]`</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 对提取出的动作值取绝对值，得到 <span style="color: #007BFF;">$|\mathbf{a}_{hip\_yaw}|$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iv. 沿着指定维度（<span style="color: #007BFF;">`dim=1`</span>）对绝对值后的动作值进行求和，得到 <span style="color: #007BFF;">$R = \sum_{i} |a_{hip\_yaw, i}|$</span>。</span>  
    该函数用于对髋关节偏航关节的动作进行惩罚，鼓励机器人减少不必要的髋关节偏航动作。

23. <span style="font-size: 1.2em; font-weight: bold;">`feet_y_distance`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 计算左脚相对于根节点的位置 <span style="color: #007BFF;">$\mathbf{p}_{left\_rel} = \mathbf{p}_{left} - \mathbf{p}_{root}$</span>，其中 <span style="color: #007BFF;">$\mathbf{p}_{left}$</span> 是左脚位置，<span style="color: #007BFF;">$\mathbf{p}_{root}$</span> 是根节点位置。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 计算右脚相对于根节点的位置 <span style="color: #007BFF;">$\mathbf{p}_{right\_rel} = \mathbf{p}_{right} - \mathbf{p}_{root}$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 将左脚位置转换到根节点坐标系下，<span style="color: #007BFF;">$\mathbf{p}_{left\_b} = \text{quat\_apply}(\text{quat\_conjugate}(\mathbf{q}_{root}), \mathbf{p}_{left\_rel})$</span>，其中 <span style="color: #007BFF;">$\mathbf{q}_{root}$</span> 是根节点四元数。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iv. 将右脚位置转换到根节点坐标系下，<span style="color: #007BFF;">$\mathbf{p}_{right\_b} = \text{quat\_apply}(\text{quat\_conjugate}(\mathbf{q}_{root}), \mathbf{p}_{right\_rel})$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">v. 计算双脚在根节点坐标系下 y 轴的距离与目标距离的差值的绝对值 <span style="color: #007BFF;">$d_y = |p_{left\_b, y} - p_{right\_b, y} - 0.299|$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">vi. 判断命令的 y 轴速度是否较低，得到布尔向量 <span style="color: #007BFF;">$\mathbf{v}_{flag}$</span>，<span style="color: #007BFF;">$v_{flag, j} = |v_{cmd, j, 1}| < 0.1$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">vii. 计算双脚在 y 轴上距离的惩罚值 <span style="color: #007BFF;">$R = d_y \cdot \mathbf{v}_{flag}$</span>。</span>  
    该函数在命令的 y 轴速度较低时，对双脚在 y 轴上的距离进行惩罚，以保持合理间距。

## Periodic gait-based reward function
24. <span style="font-size: 1.2em; font-weight: bold;">`gait_clock`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 定义摆动相标志 <span style="color: #007BFF;">$\mathbf{s}_{swing}$</span>：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$\mathbf{s}_{swing} = (\mathbf{phase} \geq \delta_t) \land (\mathbf{phase} \leq (\mathbf{air\_ratio} - \delta_t))$</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 定义支撑相标志 <span style="color: #007BFF;">$\mathbf{s}_{stand}$</span>：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$\mathbf{s}_{stand} = (\mathbf{phase} \geq (\mathbf{air\_ratio} + \delta_t)) \land (\mathbf{phase} \leq (1 - \delta_t))$</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 定义过渡阶段标志 <span style="color: #007BFF;">$\mathbf{s}_{trans1}$</span>、<span style="color: #007BFF;">$\mathbf{s}_{trans2}$</span>、<span style="color: #007BFF;">$\mathbf{s}_{trans3}$</span>：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$\mathbf{s}_{trans1} = \mathbf{phase} < \delta_t$</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$\mathbf{s}_{trans2} = (\mathbf{phase} > (\mathbf{air\_ratio} - \delta_t)) \land (\mathbf{phase} < (\mathbf{air\_ratio} + \delta_t))$</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$\mathbf{s}_{trans3} = \mathbf{phase} > (1 - \delta_t)$</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iv. 计算摆动相时钟信号 <span style="color: #007BFF;">$\mathbf{I}_{frc}$</span>：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$\mathbf{I}_{frc} = 1.0 \cdot \mathbf{s}_{swing} + (0.5 + \frac{\mathbf{phase}}{2 \delta_t}) \cdot \mathbf{s}_{trans1} - \frac{\mathbf{phase} - \mathbf{air\_ratio} - \delta_t}{2.0 \delta_t} \cdot \mathbf{s}_{trans2} + 0.0 \cdot \mathbf{s}_{stand} + \frac{\mathbf{phase} - 1 + \delta_t}{2 \delta_t} \cdot \mathbf{s}_{trans3}$</span>  
    <span style="font-size: 1.1em; font-weight: bold;">v. 计算支撑相时钟信号 <span style="color: #007BFF;">$\mathbf{I}_{spd}$</span>：</span>  
    <span style="font-size: 1.1em; font-weight: bold;">$\mathbf{I}_{spd} = 1.0 - \mathbf{I}_{frc}$</span>  
    该函数生成用于脚摆动相和支撑相的周期性步态时钟信号，在摆动相和支撑相之间进行平滑过渡。

25. <span style="font-size: 1.2em; font-weight: bold;">`gait_feet_frc_perio`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 获取左脚摆动相的掩码 <span style="color: #007BFF;">$\mathbf{m}_{left\_frc}$</span>，<span style="color: #007BFF;">$\mathbf{m}_{left\_frc} = \text{gait\_clock}(\mathbf{phase}_{left}, \mathbf{air\_ratio}_{left}, \delta_t)[0]$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 获取右脚摆动相的掩码 <span style="color: #007BFF;">$\mathbf{m}_{right\_frc}$</span>，<span style="color: #007BFF;">$\mathbf{m}_{right\_frc} = \text{gait\_clock}(\mathbf{phase}_{right}, \mathbf{air\_ratio}_{right}, \delta_t)[0]$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 计算左脚在摆动相的脚力得分 <span style="color: #007BFF;">$\mathbf{r}_{left\_frc}$</span>，<span style="color: #007BFF;">$\mathbf{r}_{left\_frc} = \mathbf{m}_{left\_frc} \cdot \exp(-200 \cdot \mathbf{F}_{left}^2)$</span>，其中 <span style="color: #007BFF;">$\mathbf{F}_{left}$</span> 是左脚平均脚力。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iv. 计算右脚在摆动相的脚力得分 <span style="color: #007BFF;">$\mathbf{r}_{right\_frc}$</span>，<span style="color: #007BFF;">$\mathbf{r}_{right\_frc} = \mathbf{m}_{right\_frc} \cdot \exp(-200 \cdot \mathbf{F}_{right}^2)$</span>，其中 <span style="color: #007BFF;">$\mathbf{F}_{right}$</span> 是右脚平均脚力。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">v. 计算最终奖励 <span style="color: #007BFF;">$R = \mathbf{r}_{left\_frc} + \mathbf{r}_{right\_frc}$</span>。</span>  
    该函数在步态的摆动相期间惩罚脚受到的力，脚力越大奖励越小。

26. <span style="font-size: 1.2em; font-weight: bold;">`gait_feet_spd_perio`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 获取左脚支撑相的掩码 <span style="color: #007BFF;">$\mathbf{m}_{left\_spd}$</span>，<span style="color: #007BFF;">$\mathbf{m}_{left\_spd} = \text{gait\_clock}(\mathbf{phase}_{left}, \mathbf{air\_ratio}_{left}, \delta_t)[1]$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 获取右脚支撑相的掩码 <span style="color: #007BFF;">$\mathbf{m}_{right\_spd}$</span>，<span style="color: #007BFF;">$\mathbf{m}_{right\_spd} = \text{gait\_clock}(\mathbf{phase}_{right}, \mathbf{air\_ratio}_{right}, \delta_t)[1]$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 计算左脚在支撑相的脚速得分 <span style="color: #007BFF;">$\mathbf{r}_{left\_spd}$</span>，<span style="color: #007BFF;">$\mathbf{r}_{left\_spd} = \mathbf{m}_{left\_spd} \cdot \exp(-100 \cdot \mathbf{v}_{left}^2)$</span>，其中 <span style="color: #007BFF;">$\mathbf{v}_{left}$</span> 是左脚平均脚速。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iv. 计算右脚在支撑相的脚速得分 <span style="color: #007BFF;">$\mathbf{r}_{right\_spd}$</span>，<span style="color: #007BFF;">$\mathbf{r}_{right\_spd} = \mathbf{m}_{right\_spd} \cdot \exp(-100 \cdot \mathbf{v}_{right}^2)$</span>，其中 <span style="color: #007BFF;">$\mathbf{v}_{right}$</span> 是右脚平均脚速。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">v. 计算最终奖励 <span style="color: #007BFF;">$R = \mathbf{r}_{left\_spd} + \mathbf{r}_{right\_spd}$</span>。</span>  
    该函数在步态的支撑相期间惩罚脚的速度，脚速越大奖励越小。

27. <span style="font-size: 1.2em; font-weight: bold;">`gait_feet_frc_support_perio`</span>  
    <span style="font-size: 1.1em; font-weight: bold;">i. 获取左脚支撑相的掩码 <span style="color: #007BFF;">$\mathbf{m}_{left\_frc\_support}$</span>，<span style="color: #007BFF;">$\mathbf{m}_{left\_frc\_support} = \text{gait\_clock}(\mathbf{phase}_{left}, \mathbf{air\_ratio}_{left}, \delta_t)[1]$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">ii. 获取右脚支撑相的掩码 <span style="color: #007BFF;">$\mathbf{m}_{right\_frc\_support}$</span>，<span style="color: #007BFF;">$\mathbf{m}_{right\_frc\_support} = \text{gait\_clock}(\mathbf{phase}_{right}, \mathbf{air\_ratio}_{right}, \delta_t)[1]$</span>。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iii. 计算左脚在支撑相的脚支撑力得分 <span style="color: #007BFF;">$\mathbf{r}_{left\_frc\_support}$</span>，<span style="color: #007BFF;">$\mathbf{r}_{left\_frc\_support} = \mathbf{m}_{left\_frc\_support} \cdot (1 - \exp(-10 \cdot \mathbf{F}_{left}^2))$</span>，其中 <span style="color: #007BFF;">$\mathbf{F}_{left}$</span> 是左脚平均脚力。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">iv. 计算右脚在支撑相的脚支撑力得分 <span style="color: #007BFF;">$\mathbf{r}_{right\_frc\_support}$</span>，<span style="color: #007BFF;">$\mathbf{r}_{right\_frc\_support} = \mathbf{m}_{right\_frc\_support} \cdot (1 - \exp(-10 \cdot \mathbf{F}_{right}^2))$</span>，其中 <span style="color: #007BFF;">$\mathbf{F}_{right}$</span> 是右脚平均脚力。</span>  
    <span style="font-size: 1.1em; font-weight: bold;">v. 计算最终奖励 <span style="color: #007BFF;">$R = \mathbf{r}_{left\_frc\_support} + \mathbf{r}_{right\_frc\_support}$</span>。</span>  