import numpy as np

import os
import hashlib
import json
import h5py
from datetime import datetime
from scipy.spatial.transform import Rotation



##############################
## generic variables
##############################
# define tensor_Levi_Civita
tensor_Levi_Civita = np.zeros((3,3,3))
tensor_Levi_Civita[0, 1, 2] = tensor_Levi_Civita[1, 2, 0] = tensor_Levi_Civita[2, 0, 1] = 1
tensor_Levi_Civita[0, 2, 1] = tensor_Levi_Civita[1, 0, 2] = tensor_Levi_Civita[2, 1, 0] = -1



##############################
## generic functions
##############################
# Calculate distance matrix from coordinates
def calculate_distance_matrix(coords, I=None, J=None, switch_bsx=True):
    """
    计算坐标矩阵中所有点之间的欧氏距离矩阵
    
    参数:
        coords    : numpy数组，形状为(N, 3)，表示N个三维坐标点
        I         : 整数/列表/None，指定行索引，默认为None表示所有行
        J         : 整数/列表/None，指定列索引，默认为None表示所有列
        switch_bsx: bool，是否使用广播优化，默认为True
    
    返回:
        根据输入参数返回完整矩阵、行向量、列向量或标量
    
    示例:
        >>> coords = np.array([[1,2,3], [4,5,6]])
        >>> calculate_distance_matrix(coords)
        array([[0.        , 5.19615242],
               [5.19615242, 0.        ]])
    """
    # 输入校验
    if coords.size == 0:
        return np.array([])
    
    # 参数默认值处理
    if J is None:
        J = 'ALL'
    if I is None:
        I = 'ALL'
    
    # 矩阵法计算距离矩阵
    M_ij = coords @ coords.T  # 内积矩阵
    diag_vec = np.diag(M_ij)
    
    if switch_bsx:
        # 使用广播优化计算
        distance_matrix = np.sqrt(diag_vec[:, np.newaxis] + diag_vec[np.newaxis, :] - 2 * M_ij)
    else:
        # 备用计算方法（等效于MATLAB的bsxfun）
        distance_matrix = np.sqrt(np.add.outer(diag_vec, diag_vec) - 2 * M_ij)
    
    # 处理不同返回形式
    if J == 'ALL':
        if I == 'ALL':
            return distance_matrix
        else:
            return distance_matrix[:, I]
    else:
        return distance_matrix[J, I]


# Calculate rotation between two vectors
def vrrotvec(a, b, epsilon=1e-12):
    """
    计算从向量a到向量b的旋转轴和角度（模拟MATLAB vrrotvec）
    返回: [axis_x, axis_y, axis_z, angle_rad]
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # 处理零向量
    if np.linalg.norm(a) < epsilon or np.linalg.norm(b) < epsilon:
        return np.array([0., 0., 1., 0.])  # 零向量默认返回无旋转
    
    # 单位化向量
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    
    # 计算旋转轴（叉乘）
    v = np.cross(a_norm, b_norm)
    v_norm = np.linalg.norm(v)
    
    # 处理共线情况
    if v_norm < epsilon:
        # 检查是否反向
        if np.dot(a_norm, b_norm) < -1 + epsilon:
            # 寻找垂直于a的任意轴
            if abs(a_norm[0]) > epsilon or abs(a_norm[1]) > epsilon:
                axis = np.array([a_norm[1], -a_norm[0], 0])
            else:
                axis = np.array([0, a_norm[2], -a_norm[1]])
            axis = axis / np.linalg.norm(axis)
            return np.append(axis, np.pi)
        else:
            return np.array([0., 0., 1., 0.])  # 无旋转
    
    # 计算旋转角度
    angle = np.arctan2(v_norm, np.dot(a_norm, b_norm))
    
    # 规范化旋转轴
    axis = v / v_norm
    return np.append(axis, angle)


# Convert rotation from axis-angle to matrix representation
def vrrotvec2mat(rotvec, epsilon=1e-12):
    """
    将轴角向量转换为旋转矩阵（四元素输入：[axis, angle]）
    """
    if np.linalg.norm(rotvec[:3]) < epsilon:
        return np.eye(3)  # 无旋转
    
    # 创建旋转对象
    rotation = Rotation.from_rotvec(rotvec[3] * rotvec[:3])
    return rotation.as_matrix()


# Get values and indices of extremes
def get_extremes(data, extreme_type="min", threshold=1e-8, print_result=False):
    if extreme_type.lower() == "min":
        extreme_value = np.nanmin(data)
        extreme_label = "Minimal"
    else:
        extreme_value = np.nanmax(data)
        extreme_label = "Maximal"
    
    print("{:} value = {:.8f}".format(extreme_label, extreme_value))
    
    extreme_idx = np.where(np.abs(data-extreme_value) < threshold)[0]
    if print_result:
        print("Indices of {:} values [{:d}]:".format(extreme_label.lower(), len(extreme_idx)))
        for idx in extreme_idx:
            print("Step {:08d}: {:.6f}".format(idx,data[idx]))
    else:
        print("Number of {:} values: [{:d}]".format(extreme_label.lower(), len(extreme_idx)))
    
    return extreme_idx, extreme_value


# Compute Angular Momentum
def compute_angular_momentum(r, r_dot, mass):
    """
    计算系统总角动量
    参数:
        r: 位置矩阵，形状 (3,3,N) [质点, 坐标, 时间步]
        r_dot: 速度矩阵，形状同 r
        mass: 质量向量，形状 (3,)
    返回:
        总角动量，形状 (3, N) [x/y/z分量, 时间步]
    """
    # 计算动量 p = mass * velocity
    p = mass[:, np.newaxis, np.newaxis] * r_dot  # 广播到 (3,3,N)
    
    # 使用爱因斯坦求和计算角动量
    return np.einsum('jkl,ikb,ilb->jb', tensor_Levi_Civita, r, p, optimize='optimal')

# Compute Inertia Tensor
# I = \sum_{i=1}^N {m_i ( r_i^c r_i^c \delta^{ab} - r_i^a r_i^b )}
def compute_inertia_tensor(r, mass):
    """
    计算转动惯量张量
    参数:
        r: 位置矩阵，形状 (num_particles, 3, time_steps)
           例如 (3,3,1000) 表示3个质点、3个坐标轴、1000个时间步
        mass: 质量向量，形状 (num_particles,)
    返回:
        转动惯量张量，形状 (3, 3, time_steps)
    """
    # 构造克罗内克delta张量（单位矩阵）
    delta = np.eye(3)  # 形状 (3,3)
    
    # 第一部分：Σ m_i * |r_i|² * I_3
    part1 = np.einsum('ab,i,ict,ict->abt', delta, mass, r, r)
    
    # 第二部分：-Σ m_i * r_i ⊗ r_i
    part2 = np.einsum('i,iat,ibt->abt', mass, r, r)
    
    # 总转动惯量张量
    I = part1 - part2
    return I


# Compute Angular Velocity
# inertia_tensor @ angular_velocity = angular_momentum
def compute_angular_velocity(inertia_tensor, angular_momentum):
    """
    从转动惯量张量和角动量计算角速度矢量
    参数:
        inertia_tensor  : 转动惯量张量，形状 (3,3,N)
        angular_momentum: 角动量矢量，形状 (3,N)
    返回:
        omega: 角速度矢量，形状 (3,N)
    """
    # 检查输入维度
    assert inertia_tensor.shape == (3, 3, angular_momentum.shape[1]), f"维度不匹配: inertia_tensor {inertia_tensor.shape}, angular_momentum {angular_momentum.shape}"
    assert angular_momentum.shape[0] == 3, f"角动量维度错误: angular_momentum 应为 (3, N)，实际为 {angular_momentum.shape}"
    
    # 调整维度以匹配批量求解要求
    inertia_tensor_reshaped = np.moveaxis(inertia_tensor, 2, 0)      # (N,3,3)
    angular_momentum_reshaped = angular_momentum.T[:, :, np.newaxis] # (N,3,1)
    
    # 批量求解线性方程组 I * omega = L
    angular_velocity_reshaped = np.linalg.solve(inertia_tensor_reshaped, angular_momentum_reshaped)
    angular_velocity = np.squeeze(angular_velocity_reshaped, axis=2).T  # (3,N)
    
    return angular_velocity


##############################
## Jacobi coordinates
##############################
# shape coordinates -> Jacobi coordinates
def shape_to_Jacobi_coordinates(S,mass):
    # reduece masses
    mu = [mass[0]*mass[2]/(mass[0]+mass[2]), (mass[0]+mass[2])*mass[1]/(mass[0]+mass[1]+mass[2])]
    # Jacobi Coordinates
    rho = np.full((2,3,S.shape[-1]), np.nan)
    #
    rho[0,:,:] = np.sqrt(mu[0]) * (S[0,:,:] - S[2,:,:])
    rho[1,:,:] = np.sqrt(mu[1]) * (S[1,:,:] - (mass[0]*S[0,:,:]+mass[2]*S[2,:,:])/(mass[0]+mass[2]))
    #
    return rho

# Jacobi coordinates -> shape coordinates
def Jacobi_to_shape_coordinates(rho,mass):
    # reduece masses
    mu = [mass[0]*mass[2]/(mass[0]+mass[2]), (mass[0]+mass[2])*mass[1]/(mass[0]+mass[1]+mass[2])]
    # shape coordniates
    S = np.full((3,3,rho.shape[-1]), np.nan)
    #
    S[0,:,:] = np.sqrt(mu[0]) * rho[0,:,:] / mass[0] - np.sqrt(mu[1]) * rho[1,:,:] / (mass[0]+mass[2])
    S[1,:,:] = np.sqrt(mu[1]) * rho[1,:,:] / mass[1]
    S[2,:,:] =-np.sqrt(mu[0]) * rho[0,:,:] / mass[2] - np.sqrt(mu[1]) * rho[1,:,:] / (mass[0]+mass[2])
    #
    return S

##############################
## shape_space_transformation
##############################
def shape_space_transformation(R_in, transform, zeroThreshold=1e-12, mass=None):
    """
    Python实现MATLAB的shape_space_transformation函数
    参数:
        R_in: 3x3 numpy数组，输入坐标矩阵（每行代表一个点的xyz坐标）
        transform: 字符串，变换类型（'OXY', 'MCXY', 'XYMC', 'XMCY'）
        zeroThreshold: 零值判定阈值
        mass: 质量向量（默认全1）
    返回:
        变换后的3x3坐标矩阵
    """
    R = R_in.copy().astype(np.float64)
    N = R.shape[0]
    
    # 处理默认参数
    if mass is None:
        mass = np.ones(3)
    else:
        mass = np.asarray(mass, dtype=np.float64)
    
    if N != 3:
        raise ValueError("R.shape[0] does not equal to 3!")
        
    # 执行变换逻辑
    if transform == 'OXY':  # P1 at origin, P2 at positive x-axis, P3 in y>0 plane
        # 平移第一个点到原点
        if np.any(np.abs(R[0]) > zeroThreshold):
            R -= R[0]
        
        # 旋转第二个点到x轴正方向
        if np.linalg.norm(R[1]) > zeroThreshold and abs(R[1,1]) > zeroThreshold:
            target = np.array([np.linalg.norm(R[1]), 0, 0])
            rotvec = vrrotvec(R[1], target, epsilon=zeroThreshold)
            rotmat = vrrotvec2mat(rotvec)
            R = R @ rotmat.T
        
        # 确保第三个点在y>0平面
        if np.linalg.norm(R[2]) > zeroThreshold and R[2,1] < -zeroThreshold:
            R[2,1] *= -1
    
    elif transform == 'MCXY': # Mass center at origin, P2 at positive x-axis, P3 in y>0 plane
        # 质心移动到原点
        MC = (mass @ R) / np.sum(mass)
        if np.any(np.abs(MC) > zeroThreshold):
            R -= MC
        
        # 旋转第二个点到x轴正方向
        if np.linalg.norm(R[1]) > zeroThreshold and abs(R[1,1]) > zeroThreshold:
            target = np.array([np.linalg.norm(R[1]), 0, 0])
            rotvec = vrrotvec(R[1], target, epsilon=zeroThreshold)
            rotmat = vrrotvec2mat(rotvec)
            R = R @ rotmat.T
        
        # 调整第三个点和第一个点的y坐标
        if np.linalg.norm(R[2]) > zeroThreshold and R[2,1] < -zeroThreshold:
            R[2,1] *= -1
            R[0,1] *= -1
    
    elif transform == 'XYMC': # Mass center at origin, P1 at positive x-axis, P2 in y>0 plane 
        # 质心移动到原点
        MC = (mass @ R) / np.sum(mass)
        if np.any(np.abs(MC) > zeroThreshold):
            R -= MC
        
        # 旋转第一个点到x轴正方向
        if np.linalg.norm(R[0]) > zeroThreshold and abs(R[0,1]) > zeroThreshold:
            target = np.array([np.linalg.norm(R[0]), 0, 0])
            rotvec = vrrotvec(R[0], target, epsilon=zeroThreshold)
            rotmat = vrrotvec2mat(rotvec)
            R = R @ rotmat.T
        
        # 确保第二个点在y>0平面
        if np.linalg.norm(R[1]) > zeroThreshold and R[1,1] < -zeroThreshold:
            R[1:,1] *= -1  # 同时翻转后续点的y坐标
    
    elif transform == 'XMCY': # Mass center at origin, P1 at positive x-axis, P3 in y>0 plane   
        # 质心移动到原点
        MC = (mass @ R) / np.sum(mass)
        if np.any(np.abs(MC) > zeroThreshold):
            R -= MC
        
        # 旋转第一个点到x轴正方向
        if np.linalg.norm(R[0]) > zeroThreshold and abs(R[0,1]) > zeroThreshold:
            target = np.array([np.linalg.norm(R[0]), 0, 0])
            rotvec = vrrotvec(R[0], target, epsilon=zeroThreshold)
            rotmat = vrrotvec2mat(rotvec)
            R = R @ rotmat.T
        
        # 调整第三个点的y坐标
        if np.linalg.norm(R[2]) > zeroThreshold and R[2,1] < -zeroThreshold:
            R[1:,1] *= -1
    
    else:
        raise ValueError(f"Unknown transform type: {transform}")
    
    return R



##############################
## I/O functions
##############################
# 自定义哈希函数
def hash_run_information(in_dict, exclusion_list=['dij','S','R','Theta']):
    sorted_items = sorted([tpl for tpl in in_dict.items() if tpl[0] not in exclusion_list])
    hash_string = ''.join(f"{key}{value}" for key, value in sorted_items)
    return hashlib.sha256(hash_string.encode()).hexdigest()

def generate_savename(in_dict):
    """根据参数生成标准化文件名"""
    return "{:}_{:}_{:}".format(in_dict['runLabel'], in_dict['funcType'], hash_run_information(in_dict))


# 存储中间结果 ver_1
'''
def save_intermediate_results(results_dict, save_name=None, save_dir='./intermediate_results'):
    """
    增强版中间结果保存函数，支持多种数据类型
    
    参数:
        results_dict: 包含多种数据类型的字典，支持类型包括:
            - numpy.ndarray
            - int/float
            - str
            - list/dict（通过JSON序列化）
        save_name: 保存目录名称
        save_dir: 保存目录路径
    """
    # 创建保存目录,默认为带时间戳的目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, timestamp if save_name is None else save_name)
    os.makedirs(save_path, exist_ok=True)

    # 初始化元数据存储
    metadata = {
        'creation_time': timestamp,
        'data_info': {}
    }

    for name, data in results_dict.items():
        data_info = {'dtype': type(data).__name__}
        
        try:
            # 处理不同数据类型
            if isinstance(data, np.ndarray):
                # 保存NumPy数组
                np.save(os.path.join(save_path, f'{name}.npy'), data)
                data_info.update({
                    'format': 'npy',
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                })
                
            elif isinstance(data, (int, float)):
                # 保存标量为NumPy标量文件
                np.save(os.path.join(save_path, f'{name}.npy'), np.array(data))
                data_info['format'] = 'npy_scalar'
                
            elif isinstance(data, str):
                # 保存文本文件
                with open(os.path.join(save_path, f'{name}.txt'), 'w') as f:
                    f.write(data)
                data_info['format'] = 'txt'
                
            elif isinstance(data, (list, dict)):
                # 保存JSON文件
                with open(os.path.join(save_path, f'{name}.json'), 'w') as f:
                    json.dump(data, f)
                data_info['format'] = 'json'
                
            else:
                # 尝试序列化未知类型
                try:
                    np.save(os.path.join(save_path, f'{name}.npy'), data)
                    data_info['format'] = 'npy_unknown'
                except Exception as e:
                    raise ValueError(f"无法保存类型 {type(data)} 的数据 {name}: {str(e)}")
                    
            metadata['data_info'][name] = data_info
            
        except Exception as e:
            print(f"保存 {name} 失败: {str(e)}")
            continue

    # 保存元数据
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"数据已保存至：{save_path}")
    
    return save_path
'''

# 存储中间结果 ver_2
def save_intermediate_results(
    results_dict,
    save_name=None,
    save_dir='./intermediate_results',
    hdf5_threshold=1000000,  # 1MB阈值
    hdf5_compression= 'gzip',
    hdf5_chunks=True
):
    """
    支持HDF5的增强版中间结果保存函数
    
    参数:
        results_dict: 包含多种数据类型的字典
        save_dir: 保存目录路径
        hdf5_threshold: 使用HDF5的数组元素数量阈值
        hdf5_compression: 压缩算法 ('gzip', 'lzf', None)
        hdf5_chunks: 是否启用分块存储
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, timestamp if save_name is None else save_name)
    print(save_path)
    os.makedirs(save_path, exist_ok=True)

    metadata = {
        'creation_time': timestamp,
        'run_hash': hash_run_information(results_dict),
        'data_info': {},
        'hdf5_file': 'data.h5'  # 统一HDF5文件名
    }

    hdf5_path = os.path.join(save_path, metadata['hdf5_file'])
    hdf5_handle = h5py.File(hdf5_path, 'w')

    try:
        for name, data in results_dict.items():
            data_info = {'dtype': type(data).__name__}
            
            try:
                # 处理HDF5大型数组
                if isinstance(data, np.ndarray) and data.size >= hdf5_threshold:
                    # 计算分块参数
                    chunks = True if hdf5_chunks else None
                    if chunks and data.ndim > 0:
                        chunks = tuple([min(dim, 1000) for dim in data.shape])
                    
                    # 创建数据集
                    dset = hdf5_handle.create_dataset(
                        name,
                        data=data,
                        chunks=chunks,
                        compression=hdf5_compression
                    )
                    
                    data_info.update({
                        'format': 'hdf5',
                        'shape': list(data.shape),
                        'dtype': str(data.dtype),
                        'compression': hdf5_compression,
                        'chunks': chunks
                    })
                    
                # 处理其他数据类型
                elif isinstance(data, np.ndarray):
                    np.save(os.path.join(save_path, f'{name}.npy'), data)
                    data_info.update({
                        'format': 'npy',
                        'shape': list(data.shape),
                        'dtype': str(data.dtype)
                    })
                    
                elif isinstance(data, (int, float)):
                    np.save(os.path.join(save_path, f'{name}.npy'), np.array(data))
                    data_info['format'] = 'npy_scalar'
                    
                elif isinstance(data, str):
                    with open(os.path.join(save_path, f'{name}.txt'), 'w') as f:
                        f.write(data)
                    data_info['format'] = 'txt'
                    
                elif isinstance(data, (list, dict)):
                    with open(os.path.join(save_path, f'{name}.json'), 'w') as f:
                        json.dump(data, f)
                    data_info['format'] = 'json'
                    
                else:
                    try:
                        np.save(os.path.join(save_path, f'{name}.npy'), data)
                        data_info['format'] = 'npy_unknown'
                    except Exception as e:
                        raise ValueError(f"无法保存类型 {type(data)} 的数据 {name}: {str(e)}")
                        
                metadata['data_info'][name] = data_info
                
            except Exception as e:
                print(f"保存 {name} 失败: {str(e)}")
                continue
                
    finally:
        hdf5_handle.close()

    # 保存元数据
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"数据已保存至：{save_path}")
    
    return save_path


# 读取中间结果 ver_1
'''
def load_intermediate_results(load_path):
    """
    加载增强版中间结果
    
    参数:
        load_path: 保存目录路径
    
    返回:
        包含加载数据的字典
    """
    results = {}
    
    # 加载元数据
    with open(os.path.join(load_path, 'metadata.json')) as f:
        metadata = json.load(f)
    
    for name, info in metadata['data_info'].items():
        try:
            file_path = os.path.join(load_path, f"{name}.{info['format'].split('_')[0]}")
            
            if info['format'].startswith('npy'):
                data = np.load(file_path, allow_pickle=True)
                if info['format'] == 'npy_scalar':
                    data = data.item()  # 转换为Python标量
                results[name] = data
                
            elif info['format'] == 'txt':
                with open(file_path) as f:
                    results[name] = f.read()
                    
            elif info['format'] == 'json':
                with open(file_path) as f:
                    results[name] = json.load(f)
                    
        except Exception as e:
            print(f"加载 {name} 失败: {str(e)}")
            continue
    
    return results
'''

# 读取中间结果 ver_2
def load_intermediate_results(load_path):
    """
    加载支持HDF5的中间结果
    
    参数:
        load_path: 保存目录路径
    
    返回:
        包含加载数据的字典
    """
    results = {}
    
    # 加载元数据
    with open(os.path.join(load_path, 'metadata.json')) as f:
        metadata = json.load(f)
    
    # 打开HDF5文件
    hdf5_file = os.path.join(load_path, metadata['hdf5_file'])
    hdf5_handle = h5py.File(hdf5_file, 'r') if os.path.exists(hdf5_file) else None
    
    try:
        for name, info in metadata['data_info'].items():
            try:
                if info['format'] == 'hdf5' and hdf5_handle:
                    # 从HDF5加载数据集
                    if name in hdf5_handle:
                        data = hdf5_handle[name][()]
                        if info['dtype'] == 'ndarray':
                            results[name] = np.array(data)
                        else:
                            results[name] = data
                    else:
                        print(f"警告：HDF5数据集 {name} 不存在")
                        
                else:
                    # 原有加载逻辑
                    file_path = os.path.join(load_path, f"{name}.{info['format'].split('_')[0]}")
                    
                    if info['format'].startswith('npy'):
                        data = np.load(file_path, allow_pickle=True)
                        if info['format'] == 'npy_scalar':
                            data = data.item()
                        results[name] = data
                        
                    elif info['format'] == 'txt':
                        with open(file_path) as f:
                            results[name] = f.read()
                            
                    elif info['format'] == 'json':
                        with open(file_path) as f:
                            results[name] = json.load(f)
                            
            except Exception as e:
                print(f"加载 {name} 失败: {str(e)}")
                continue
                
    finally:
        if hdf5_handle:
            hdf5_handle.close()
    
    return results



##############################
## visualization functions
##############################
# plot triangle
def plot_triangle(points, ax, color, label, 
                  linestyle='--', linewidth=1.5, 
                  show_marker=False, markersize=8, markerfacecolor=None, markeredgecolor='k',
                  show_text=True, text_prefix='P', text_fontsize=10, text_color=None, text_offset=[[0.05, 0.05]],
                  show_edge=True, edge_color=None, edge_alpha=0.8, edge_offset=0.15):
    """
    绘制三角形并添加标签
    """
    
    if show_marker:
        # 连接三个点形成闭合三角形
        closed_points = np.vstack([points, points[0]])
        ax.plot(closed_points[:,0], closed_points[:,1], 
                color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        # 三角形三个顶点
        for i, (x, y, _) in enumerate(points):
            if len(markerfacecolor) == len(points):
                mf_color = markerfacecolor[i]
            elif len(text_color) == 1:
                mf_color = markerfacecolor
            else:
                mf_color = color
            ax.plot(x, y, marker='o', markersize=markersize,
                    markerfacecolor=mf_color, markeredgecolor=markeredgecolor)
    else:
        # 连接三个点形成闭合三角形
        closed_points = np.vstack([points, points[0]])
        ax.plot(closed_points[:,0], closed_points[:,1], 
                color=color, marker='o', linestyle=linestyle,
                linewidth=linewidth, markersize=markersize, label=label)
        
    # 添加点标签
    if show_text:
        for i, (x, y, _) in enumerate(points):
            if len(text_color) == len(points):
                t_color = text_color[i]
            elif len(text_color) == 1:
                t_color = text_color
            else:
                t_color = color
            if len(text_offset) == len(points):
                t_offset = text_offset[i]
            elif len(text_offset) == 1:
                t_offset = text_offset[0]
            else:
                t_offset = [0,0]
            ax.text(x+t_offset[0], y+t_offset[1], "{:}{:d}".format(text_prefix,i+1),
                    fontsize=text_fontsize, color=t_color, ha='left', va='bottom')

    # 计算边信息
    if show_edge:
        edges = [(0, 1), (1, 2), (2, 0)]  # [P1-P2, P2-P3, P3-P1] 
        edge_labels = []
        for (i, j) in edges:
            # 获取端点坐标
            p1, p2 = points[i], points[j]

            # 计算中点坐标
            mid_point = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
            # 计算边长
            length = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            # 计算标签偏移方向（垂直边方向）
            dx = p2[1] - p1[1]
            dy = p1[0] - p2[0]
            direction = np.array([dx, dy]) / np.sqrt(dx**2 + dy**2)

            # 存储边信息
            edge_labels.append({
                'position': mid_point,
                'length': length,
                'direction': direction
            })

        # 添加边长标签
        label_style = {
            'fontsize': 10,
            'color': color if edge_color is None else edge_color,
            'ha': 'center',
            'va': 'center',
            'bbox': dict(facecolor='white', alpha=edge_alpha, edgecolor='none')
        }

        for edge in edge_labels:
            # 计算标签位置（向垂直方向偏移）
            offset = edge_offset * edge['direction']
            text_pos = [edge['position'][0] + offset[0], edge['position'][1] + offset[1]]
            # 格式化长度显示
            length_text = f"{edge['length']:.2f}"
            # 添加文本
            ax.text(*text_pos, length_text, **label_style)


# visualization
def plot_visualization(S, R, dij, Thetadot, Theta, ResDiff, steps, deltaT, mass, funcType, varType):
    """
    执行所有可视化任务
    
    参数:
        S: 形状空间坐标 (3,3,T)
        R: 真实空间坐标 (3,3,T)
        dij: 距离矩阵 (3,T)
        Thetadot: 角速度 (T,)
        Theta: 角度 (T,)
        ResDiff: 残差差异 (T,)
        steps: 时间步数组
        deltaT: 时间步长
        mass: 质量数组
        funcType: 函数类型 ('sin'/'tanh')
        varType: 变量类型
    """
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2,
        'legend.fontsize': 12
    })

    # ==================================================================
    # Part 1: 距离矩阵可视化
    # ==================================================================
    def plot_distance_matrix():
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        edge_labels = ['12', '13', '23']
        colors = ['#2ca02c', '#d62728', '#1f77b4']  # 与MATLAB颜色对应
        
        for i in range(3):
            ax = axs[i]
            ax.plot(steps*deltaT, dij[i], '-o', color=colors[i], markersize=4)
            ax.set_ylabel(f'$d_{{{edge_labels[i]}}}$', rotation=0, ha='right', va='center')
            ax.grid(True, linestyle=':')
            
            if varType.endswith("Pub"):
                ax.tick_params(axis='both', which='major', labelsize=24)
                ax.yaxis.label.set_size(32)
                if i == 2:
                    ax.set_xlabel('Time', fontsize=32)
            else:
                if i != 2:
                    ax.set_xticklabels([])
        
        plt.tight_layout()
        plt.savefig('distance_matrix.png', dpi=300)
        plt.close()

    # ==================================================================
    # Part 2: 振幅和角度可视化
    # ==================================================================
    def plot_amplitudes_and_angles():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 振幅图
        colors = ['#ff7f0e', '#9467bd', '#8c564b']
        for i in range(3):
            ax1.plot(steps*deltaT, dij[i], color=colors[i], label=f'$a_{{{i+1}}}$')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        
        # 角度图
        ax2.plot(steps[1:]*deltaT, Thetadot, 'r-', label='$dθ/dt$')
        ax2.plot(steps*deltaT, Theta, 'b-', label='$θ$')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Angle')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('amplitudes_angles.png', dpi=300)
        plt.close()

    # ==================================================================
    # Part 3: 轨迹可视化
    # ==================================================================
    def plot_trajectories():
        fig = plt.figure(figsize=(20, 10))
        
        # 形状空间
        ax1 = fig.add_subplot(121, projection='3d')
        for i in range(3):
            ax1.plot(S[i,0], S[i,1], S[i,2], 
                    color=plt.cm.tab10(i), 
                    linewidth=2,
                    label=f'P{i+1}')
        ax1.set_title('Shape Space')
        ax1.view_init(elev=20, azim=45)
        
        # 真实空间
        ax2 = fig.add_subplot(122, projection='3d')
        for i in range(3):
            ax2.plot(R[i,0], R[i,1], R[i,2],
                    color=plt.cm.tab10(i),
                    linewidth=2)
        ax2.set_title('Real Space')
        ax2.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig('trajectories.png', dpi=300)
        plt.close()

    # ==================================================================
    # Part 4: 生成动画
    # ==================================================================
    def generate_animation():
        metadata = dict(title='Three Body Animation', artist='Python')
        writer = FFMpegWriter(fps=30, metadata=metadata)

        fig = plt.figure(figsize=(20, 10))
        
        with writer.saving(fig, "three_body_animation.mp4", dpi=100):
            for t in range(0, S.shape[2], 100):  # 每100帧采样一次
                plt.clf()
                
                # 形状空间
                ax1 = fig.add_subplot(121, projection='3d')
                for i in range(3):
                    ax1.plot(S[i,0,:t], S[i,1,:t], S[i,2,:t], 
                            color=plt.cm.tab10(i),
                            alpha=0.5)
                    ax1.scatter(S[i,0,t], S[i,1,t], S[i,2,t],
                              color=plt.cm.tab10(i),
                              s=100)
                ax1.set_title(f'Shape Space (t={t*deltaT:.2f}s)')
                ax1.view_init(elev=20, azim=45)
                
                # 真实空间
                ax2 = fig.add_subplot(122, projection='3d')
                for i in range(3):
                    ax2.plot(R[i,0,:t], R[i,1,:t], R[i,2,:t],
                            color=plt.cm.tab10(i),
                            alpha=0.5)
                    ax2.scatter(R[i,0,t], R[i,1,t], R[i,2,t],
                              color=plt.cm.tab10(i),
                              s=100)
                ax2.set_title(f'Real Space (t={t*deltaT:.2f}s)')
                ax2.view_init(elev=20, azim=45)
                
                plt.tight_layout()
                writer.grab_frame()

    # ==================================================================
    # 执行所有绘图任务
    # ==================================================================
    plot_distance_matrix()
    plot_amplitudes_and_angles()
    plot_trajectories()
    generate_animation()