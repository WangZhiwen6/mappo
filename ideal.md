# 神经网络

现在是2个lstm网络，一个actor结合， 一个critic结合

所有智能体共用同一个 actor LSTM 网络参数
每个智能体各自维护一份 hidden/cell state

多区域建筑中暖通空调系统（HVAC）的低层控制优化（直接控制VAV末端送风流量和新风阀门）

# 超启发式算法

研究一下和启发式算法的结合进行奖励权重自适应 

启发式算法，简单来说，就是基于经验、直观或试错法的算法，它不追求找到问题的精确最优解，而是在可接受的时间内，寻找一个足够好的近似解 。


# 新风比例
在 EnergyPlus 中，室外风控制通常通过 Outdoor Air Controller 的 Air Mass Flow Rate actuator 实现；因此所谓新风比例调节，在官方接口层面更接近对新风质量流量的直接控制，而不是显式的风阀开度百分比控制。

“新风比例”在 EnergyPlus 里通常是结果，不是直接的 actuator 名称。

用 Air System Outdoor Air Flow Fraction 做闭环反馈。