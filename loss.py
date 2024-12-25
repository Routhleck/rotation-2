import jax
import jax.numpy as jnp


def communicability_loss(weight_matrix, comms_factor=1):
    """
    计算网络的通信能力损失
    
    基于无偏加权通信能力的损失函数，参考文献：
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure 
    applied to complex brain networks. Journal of the Royal Society Interface, 6(33), 411-414.
    
    参数:
        weight_matrix: 权重矩阵
        comms_factor: 通信因子，用于调节通信矩阵的影响程度
    
    返回:
        通信损失值
    """
    # 计算权重矩阵的绝对值
    abs_weight_matrix = jnp.abs(weight_matrix)

    # 计算加权通信能力
    # 步骤1：计算每个节点的权重和
    node_strengths = jnp.sum(abs_weight_matrix, axis=1)
    # 步骤2：计算节点强度的-0.5次方
    strength_norm = jnp.pow(node_strengths, -0.5)
    # 步骤3：构建对角矩阵
    strength_diag = jnp.diag(strength_norm)
    # 步骤4：计算标准化的指数矩阵
    comms_matrix = jax.scipy.linalg.expm(strength_diag @ abs_weight_matrix @ strength_diag)
    # 步骤5：移除自连接（对角线元素）
    comms_matrix = comms_matrix.at[jnp.diag_indices(comms_matrix.shape[0])].set(0)

    # 应用通信因子
    comms_matrix = comms_matrix ** comms_factor
    
    # 计算最终的通信损失
    comms_weight_matrix = jnp.multiply(abs_weight_matrix, comms_matrix)
    
    return jnp.sum(comms_weight_matrix)
