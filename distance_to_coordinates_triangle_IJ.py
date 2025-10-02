import numpy as np
from tqdm import tqdm
from scipy.linalg import eig
from joblib import Parallel, delayed
from utils import shape_space_transformation, calculate_distance_matrix


##############################
## Parrallel Version
##############################
def process_timepoint(t, M, D, transform, zero_threshold, mass):
    """
    处理单个时间点的函数（供并行调用
    """
    # 特征分解
    Mt = M[:, :, t]
    w, v = eig(Mt)
    Lambda = np.diag(np.sqrt(np.maximum(w.real, 0)))  # 处理负特征值
    
    # 构建坐标矩阵
    X = v @ Lambda
    
    # 移除零列
    zero_cols = np.all(np.abs(X) < zero_threshold, axis=0)
    X = X[:, ~zero_cols]
    
    # 补零到3列
    if X.shape[1] < 3:
        X = np.hstack([X, np.zeros((3, 3 - X.shape[1]))])
    
    # 形状空间变换
    X_transformed = shape_space_transformation(X, transform, zero_threshold, mass)
    
    # 计算残差
    D_recovered = calculate_distance_matrix(X_transformed)
    res_diff = np.max(np.abs(D_recovered - D[:, :, t]))
    
    return X_transformed, res_diff


def distance_to_coordinates_triangle_IJ_parallel(dij, transform, mass=None, zero_threshold=10*np.finfo(float).eps, n_jobs=-1, verbose = 0):
    """
    并行版本的距离矩阵转换函数
    
    参数新增:
        n_jobs: 并行进程数，-1表示使用所有CPU核心
    """
    # 初始化参数
    if mass is None:
        mass = np.ones(3)
    else:
        mass = np.asarray(mass, dtype=np.float64)
    
    T = dij.shape[1] # number of time step

    # ===== 步骤1: 构建三维距离矩阵D =====
    D = np.full((3, 3, T), np.nan)
    for t in tqdm(range(T), desc="Calculating D", unit="step"):
        D[:, :, t] = np.array([
            0,        dij[0, t], dij[1, t],
            dij[0, t], 0,        dij[2, t],
            dij[1, t], dij[2, t], 0
        ]).reshape(3,3)

    # ===== 步骤2: 计算中间矩阵M =====
    M = np.zeros_like(D)
    for i in range(3):
        for j in range(3):
            M[i, j, :] = 0.5 * (D[0, i, :]**2 + D[0, j, :]**2 - D[i, j, :]**2)

    # ===== 步骤3: 并行处理所有时间点 =====
    S = np.zeros((3, 3, T))
    ResDiff = np.zeros(T)
    with Parallel(n_jobs=n_jobs, verbose=verbose, return_as="generator") as parallel:
        results = parallel(delayed(process_timepoint)(t, M, D, transform, zero_threshold, mass) for t in range(T))
        with tqdm(total=T, desc="Calculating S", unit="step") as pbar:
            for t, (X_t, res_t) in enumerate(results):
                S[:, :, t] = X_t
                ResDiff[t] = res_t
                pbar.update(1)

    return S, ResDiff, M, D



##############################
## Non-Parrallel Version
##############################
def distance_to_coordinates_triangle_IJ(dij, transform, mass=None, zero_threshold=10*np.finfo(float).eps):
    """
    将距离矩阵转换为三维坐标
    
    参数:
        dij      : numpy数组, 形状为(3, T)，包含T个时间点的三条边距离[d12, d13, d23]
        transform: 字符串, 坐标变换类型（'OXY', 'MCXY'等）
        mass     : numpy数组, 形状为(3,)，质点质量向量
    
    返回:
        S: 形状为(3, 3, T)的坐标矩阵
        ResDiff: 形状为(T,)的残差差异数组
        M: 中间矩阵
        D: 构建的距离矩阵
    """
    # 参数初始化
    if mass is None:
        mass = np.ones(3)
    else:
        mass = np.asarray(mass, dtype=np.float64)
    
    T = dij.shape[1] # number of time step

    # ===== 步骤1: 构建三维距离矩阵D =====
    D = np.full((3, 3, T), np.nan)
    for t in range(T):
        D[:, :, t] = np.array([
            0,        dij[0, t], dij[1, t],
            dij[0, t], 0,        dij[2, t],
            dij[1, t], dij[2, t], 0
        ]).reshape(3,3)

    # ===== 步骤2: 计算中间矩阵M =====
    M = np.zeros_like(D)
    for i in range(3):
        for j in range(3):
            M[i, j, :] = 0.5 * (D[0, i, :]**2 + D[0, j, :]**2 - D[i, j, :]**2)

    # ===== 步骤3: 特征分解和坐标恢复 =====
    S = np.full((3, 3, T), np.nan)
    ResDiff = np.zeros(T)
    
    for t in range(T):
        # 3.1 特征分解
        Mt = M[:, :, t]
        w, v = eig(Mt)
        Lambda = np.diag(np.sqrt(np.maximum(w.real, 0)))  # 处理负特征值
        
        # 3.2 构建坐标矩阵
        X = v @ Lambda
        
        # 3.3 移除零列
        zero_cols = np.all(np.abs(X) < zero_threshold, axis=0)
        X = X[:, ~zero_cols]
        
        # 补零到3列
        if X.shape[1] < 3:
            X = np.hstack([X, np.zeros((3, 3 - X.shape[1]))])
        
        # 3.4 形状空间变换
        X_transformed = shape_space_transformation(X, transform, zero_threshold, mass)
        
        # 3.5 存储结果
        S[:, :, t] = X_transformed
        
        # 3.6 计算残差
        D_recovered = calculate_distance_matrix(X_transformed)
        ResDiff[t] = np.max(np.abs(D_recovered - D[:, :, t]))

    return S, ResDiff, M, D

# 测试用例
'''
if __name__ == "__main__":
    # 测试数据：3个时间点
    dij = np.array([
        [1.0, 1.1, 1.2],  # d12
        [1.0, 1.0, 1.0],  # d13
        [1.0, 1.0, 1.0]   # d23
    ])
    
    S, ResDiff, M, D = distance_to_coordinates_triangle_IJ(
        dij=dij,
        transform='XYMC',
        mass=np.array([2.3, 2.3, 4.8])
    )
    
    print("恢复的坐标矩阵（第一个时间点）：")
    print(S[:, :, 0])
    print("\n残差差异：", ResDiff)
'''

# 性能测试示例
if __name__ == "__main__":
    # 生成测试数据（1000个时间点）
    np.random.seed(42)
    T = 100000
    dij = 1 + 0.2*np.random.rand(3, T)  # 生成1.0-1.2之间的随机距离
    
    # 运行并行版本
    S, ResDiff, M, D = distance_to_coordinates_triangle_IJ_parallel(dij, 'XYMC', n_jobs=4)
    
    # 运行结果验证
    print(f"平均残差: {np.mean(ResDiff):.2e}")
    print(f"坐标矩阵形状: {S.shape}")