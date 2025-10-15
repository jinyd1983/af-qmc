import numpy as np
import math
import random

#AF-QMC基于一个深刻的数学定理：
#对于任何与基态有重叠的初始波函数|ψ₀⟩，经过足够长的虚时间演化后必趋于基态
#lim(β→∞) e^{-βH} |ψ₀⟩ ∝ |ψ_gs⟩

class SimpleAFQMCNumpyOnly:
    """仅使用NumPy的简化AF-QMC版本"""

    def __init__(self, L=4, N=4, U=4.0, dt=0.1, n_steps=5):
        self.L = L
        self.N = N
        self.U = U
        self.dt = dt
        self.n_steps = n_steps

        # 简单的一维链哈密顿量
        self.K = np.zeros((L, L))
        for i in range(L):
            self.K[i, (i+1)%L] = -1.0
            self.K[(i+1)%L, i] = -1.0
        print("K:")
        print(self.K)

    def run_simple(self):
        """运行简化的AF-QMC"""
        print("运行简化版AF-QMC...")

        # 初始波函数（自由电子基态）
        eigvals, eigvecs = np.linalg.eigh(self.K)
        phi = eigvecs[:, :self.N//2]

        energy_history = []
        print("eigvals, eigvecs:")
        print(eigvals)
        print(eigvecs)
        print("phi:")
        print(phi)


        for step in range(self.n_steps):
            # 辅助场采样
            # 作用：Hubbard-Stratonovich变换的核心步骤。
            # 在每个格点随机生成辅助场 σ_i = ±1
            # 这相当于对每个格点的相互作用项进行离散化采样
            # 物理意义：将电子-电子相互作用转换为电子与随机辅助场的耦合

            sigma_field = np.random.choice([-1, 1], size=self.L)
            print("sigma_field:")
            print(sigma_field)
                


            # 构建电子在辅助场势能下的哈密顿
            # 作用：构建当前时间步的有效哈密顿量。
            # potential：由辅助场产生的对角势能矩阵
            # 0.5 * self.U：来自Hubbard-Stratonovich变换的系数
            # total_H：动能 + 势能 = 当前构型下的单粒子哈密顿量

            potential = np.diag(0.5 * self.U * sigma_field)
            total_H = self.K + potential
            print("potential:")
            print(potential)
            print("total_H:")
            print(total_H)


            # 构建虚时演化 的传播子
            # 作用：构建虚时间传播算符的近似。
            # 精确的传播算符是 exp(-dt * total_H)
            # 这里使用一阶泰勒展开：exp(-dt*H) ≈ 1 - dt*H
            # 这是数值上的简化，真实代码应该使用矩阵指数
            propagator = np.eye(self.L) - self.dt * total_H

            print("propagator:")
            print(propagator)

            # 波函数 演化
            # 作用：将波函数向前传播一个时间步。
            # 物理意义：|ψ(t+dt)⟩ = exp(-dt*H) |ψ(t)⟩
            # 在辅助场表象下，波函数在每个时间步随不同的随机势演化
            phi_new = propagator @ phi
            print("phi_new:")
            print(phi_new)


            # 演化后的波函数重新正交化
            # 作用：数值稳定性处理。
            # QR分解：将传播后的波函数重新正交化
            # 防止数值误差积累导致波函数列向量不再正交
            # Q[:, :self.N//2]：只保留前N//2个正交列（对应占据轨道）
            
            Q, R = np.linalg.qr(phi_new)
            phi = Q[:, :self.N//2]
            print("phi:")
            print(phi)


            # 计算能量（简化）
            # 作用：热化期判断。
            # 前半段时间让系统"热化"到基态附近
            # 后半段时间才开始测量，避免初始瞬态影响结果

            if step > self.n_steps // 2:  # 热化后
                # 简单的能量估计
                energy = np.trace(phi.T @ self.K @ phi)
                energy_history.append(energy)
                print("energy_history:")
                print("energy_history")


            # 作用：监控计算进度和收敛情况。
            # 每20步输出一次最近10步的平均能量
            # 帮助用户观察能量是否收敛

            if step % 20 == 0 and step > 0:
                avg_energy = np.mean(energy_history[-10:]) if energy_history else 0
                print(f"步骤 {step}: E ≈ {avg_energy:.4f}")


        # 作用：计算并返回最终结果。
        # 对所有热化后的测量值取平均
        # 输出最终估计的基态能量

        if energy_history:
            final_energy = np.mean(energy_history)
            print(f"\n估计的基态能量: {final_energy:.4f}")
            return final_energy

        return 0.0

# 运行示例
if __name__ == "__main__":
    print("=" * 50)
    print("AF-QMC 简单示例")
    print("=" * 50)

    # 使用简化版本（不需要scipy）
    #qmc = SimpleAFQMCNumpyOnly(L=4, N=4, U=4.0, dt=0.05, n_steps=200)
    qmc = SimpleAFQMCNumpyOnly(L=4, N=4, U=4.0, dt=1, n_steps=5)
    energy = qmc.run_simple()

    print(f"\n对于4格点Hubbard模型(U=4):")
    print(f"AF-QMC估计能量: {energy:.4f}")
    print(f"参考值(近似): 约 -4.0 ~ -5.0")

    print("\n注意: 这是一个极度简化的教学示例!")
    print("真实的AF-QMC需要处理:")
    print("1. 符号问题")
    print("2. 更精确的传播算符")
    print("3. 分支/重加权算法")
    print("4. 更好的数值稳定性处理")
