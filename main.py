import numpy as np
import os
import time

class AHP:
    def __init__(self):
        try:
            self.Z = int(input("目标层个数：\n"))
            self.A = int(input("准则层个数：\n"))
            self.B = int(input("方案层个数：\n"))
            self.Z_to_A = self.input_matrices_zhunze()  # 输入准则层矩阵
            self.B_to_A = self.input_matrices_fangan()  # 输入方案层矩阵
        except ValueError:
            print("输入无效，请确保输入的是整数。")
            exit(1)

    def input_matrices_zhunze(self):
        matrices = []
        for z in range(self.Z):
            print(f"\n请输入目标层 Z{z+1} 对应的准则层矩阵（共 {self.A} 个）：")
            current_matrices = []
            for a in range(self.A):
                matrix = self.input_single_matrix(f"准则层 A{a+1}", self.A)
                current_matrices.append(matrix)
            matrices.append(current_matrices)
        return matrices
    
    def input_matrices_fangan(self):
        matrices = []
        for z in range(self.Z):
            for a in range(self.A):
                print(f"\n请输入目标层 Z{z+1} 和准则层 A{a+1} 对应的方案层矩阵（共 {self.B} 个）：")
                current_matrices = []
                for b in range(self.B):
                    matrix = self.input_single_matrix(f"方案层 B{b+1}", self.B)
                    current_matrices.append(matrix)
                matrices.append(current_matrices)
        return matrices
    
    def input_single_matrix(self, label, size):
        while True:
            print(f"请输入 {label} 矩阵：")
            matrix = []
            for i in range(size):
                row = input(f"请输入第 {i+1} 行（用空格分隔元素）：\n").split()
                try:
                    row = list(map(float, row))
                    if len(row) != size:  # 检查是否漏项
                        print(f"每行必须有{size}个元素，请重新输入！")
                        break
                    matrix.append(row)  # 转换为浮点数
                except ValueError:
                    print("输入包含非数字字符，请重新输入！")
                    break
            else:  # 只有在没有中断时才返回矩阵
                return np.array(matrix)
            
    def examine_matrix(self):
        for z, z_matrices in enumerate(self.Z_to_A):
            print(f"\n目标层 Z{z+1} 对应的准则层矩阵：")
            for a, matrix in enumerate(z_matrices):
                print(f"\n准则层 A{a+1} 矩阵：")
                print(matrix)

        for z, z_matrices in enumerate(self.B_to_A):
            za = divmod(z, self.A)
            print(f"\n目标层 Z{za[0]+1} 和准则层 A{za[1]+1} 对应的方案层矩阵：")
            for b, matrix in enumerate(z_matrices):
                print(f"\n方案层 B{b+1} 矩阵：")
                print(matrix)

    def calculate_weight(self):
        method = input("选择权重计算方法(1:算术平均法;2:几何平均法):\n")
        self.start_time = time.time()
        weights = []
        for z in range(self.Z):
            z_weights = []
            for a in range(self.A):
                matrix = self.Z_to_A[z][a]
                if method == '1':  
                    # 算术平均法
                    normalized_matrix = matrix / matrix.sum(axis=0)
                    weight = normalized_matrix.mean(axis=1)
                elif method == '2':  
                    # 几何平均法
                    product = np.prod(matrix, axis=1)
                    geo_mean = product ** (1 / matrix.shape[1])
                    weight = geo_mean / geo_mean.sum()
                else:
                    print("无效的选择，请选择 1 或 2。")
                    return
                
                z_weights.append(weight)
            weights.append(z_weights)

        for z, z_weights in enumerate(weights):
            print(f"\n目标层 Z{z+1} 的权重：")
            for a, weight in enumerate(z_weights):
                print(f"准则层 A{a+1} 权重：{weight}")

        return weights
    
    def calculate_lmdmax(self):  # 计算λmax
        lmdmax_values = []
        for z in range(self.Z):
            z_lmdmax = []
            for a in range(self.A):
                matrix = self.Z_to_A[z][a]
                eigvals = np.linalg.eigvals(matrix)
                lmdmax = max(eigvals)  # λmax为最大特征值
                z_lmdmax.append(lmdmax)
            lmdmax_values.append(z_lmdmax)

        for z, lmdmax in enumerate(lmdmax_values):
            print(f"\n目标层 Z{z+1} 的 λmax：")
            for a, val in enumerate(lmdmax):
                print(f"准则层 A{a+1} λmax：{val}")

        return lmdmax_values
    
    def consistency_check(self):
        """一致性检验：计算CI, 查表获得RI，后计算CR"""
        RI_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59}
        CR_values = []

        # 准则层的一致性检验
        for z in range(self.Z):
            z_CR = []
            for a in range(self.A):
                matrix = self.Z_to_A[z][a]
                eigvals = np.linalg.eigvals(matrix)
                lmdmax = max(eigvals)  # λmax为最大特征值
                n = self.A  # 这是准则层的大小
                CI = (lmdmax - n) / (n - 1)  # 计算CI
                RI = RI_dict.get(n, None)  # 如果n不在RI_dict中，返回None以提醒无效维度
                if RI is None:
                    print(f"无法找到对应维度 {n} 的RI值,请检查输入矩阵的维度。")
                    return
                CR = CI / RI if RI > 0 else 0  # 计算CR
                z_CR.append(CR)
                print(f"目标层 Z{z + 1} 和准则层 A{a + 1} 的一致性比率 CR:{CR}")
            CR_values.append(z_CR)

        # 方案层的一致性检验
        for z in range(self.Z):
            for a in range(self.A):
                za = divmod(z, self.A)
                print(f"\n目标层 Z{za[0]+1} 和准则层 A{za[1]+1} 对应的方案层矩阵：")
                z_CR = []
                for b in range(self.B):
                    matrix = self.B_to_A[z][a][b]
                    try:
                        eigvals = np.linalg.eigvals(matrix)
                        lmdmax = max(eigvals)  # λmax为最大特征值
                        n = self.B  # 这是方案层的大小
                        CI = (lmdmax - n) / (n - 1)  # 计算CI
                        RI = RI_dict.get(n, None)  # 如果n不在RI_dict中，返回None以提醒无效维度
                        if RI is None:
                            print(f"无法找到对应维度 {n} 的RI值,请检查输入矩阵的维度。")
                            return
                        CR = CI / RI if RI > 0 else 0  # 计算CR
                        z_CR.append(CR)
                        print(f"目标层 Z{za[0]+1} 和准则层 A{za[1]+1} 方案层 B{b+1} 的一致性比率 CR:{CR}")
                    except np.linalg.LinAlgError:
                        print(f"矩阵 {matrix} 不可逆，跳过该矩阵。")
                        break

                CR_values.append(z_CR)

        return CR_values

    def calculate_final_weights(self, weights):
        """计算最终权重"""
        final_weights = []  # 用于保存最终权重
        for z in range(self.Z):  # 遍历目标层
            z_final_weight = np.zeros(self.B)  # 初始化目标层对应的最终权重数组
            for a in range(self.A):  # 遍历准则层
                # 确保索引在有效范围内
                if z < len(self.B_to_A) and a < len(self.B_to_A[z]):
                    matrix = self.B_to_A[z][a]  # 获取当前目标层和准则层对应的方案层矩阵
                    criterion_weight = weights[z][a]  # 获取当前准则层的权重
                    try:
                        z_final_weight += criterion_weight @ matrix  # 使用矩阵乘法计算加权
                    except ValueError:
                        print(f"矩阵 {matrix} 和权重 {criterion_weight} 不兼容，跳过该矩阵。")
                        continue
                else:
                    print(f"警告：目标层 Z{z+1} 或准则层 A{a+1} 的矩阵未定义，跳过。")
                    continue

            final_weights.append(z_final_weight)  # 将当前目标层的最终权重添加到 final_weights
        return final_weights


    def find_optimal_solution(self, final_weights):
        """找到每个目标层的最优方案"""
        optimal_solution = []
        for z_weights in final_weights:
            best_alternative_index = np.argmax(z_weights)  # 找到最大权重的方案索引
            optimal_solution.append(best_alternative_index)
        return optimal_solution

    def output_results(self, weights, CR_values, final_weights, optimal_solution):
        """将结果输出到文件"""
        output_path = os.path.abspath("AHP_results.txt")  # 获取文件的绝对路径
        self.end_time = time.time()
        
        with open(output_path, 'w') as file:
            file.write(f"权重计算方法：\n")
            file.write(f"计算时间：{self.end_time - self.start_time:.5f}秒\n\n")
            file.write(f"一致性检验：\n")
            for z, cr in enumerate(CR_values):
                file.write(f"目标层 Z{z+1} 一致性比率 CR:\n")
                for a, val in enumerate(cr):
                    file.write(f"准则层 A{a+1} CR: {val:.4f}\n")

            file.write("\n权重结果：\n")
            for z, z_weights in enumerate(weights):
                file.write(f"目标层 Z{z+1} 权重：\n")
                for a, weight in enumerate(z_weights):
                    weight_str = np.array2string(weight, precision=2, suppress_small=True, separator=' ', floatmode='fixed')
                    file.write(f"准则层 A{a+1} 权重：\n{weight_str}\n")

            file.write("\n最终权重：\n")
            for z, z_final_weight in enumerate(final_weights):
                final_weight_str = np.array2string(z_final_weight, precision=2, suppress_small=True, separator=' ', floatmode='fixed')
                file.write(f"目标层 Z{z+1} 最终权重：\n{final_weight_str}\n")

            file.write("\n最优方案：\n")
            for z, optimal in enumerate(optimal_solution):
                file.write(f"目标层 Z{z+1} 最优方案：B{optimal+1}\n")
        
        print(f"结果已输出至: {output_path}")
        print(f"计算时间：{self.end_time - self.start_time:.5f}秒")
        
if __name__ == "__main__":
    ahp = AHP()
    ahp.examine_matrix()  # 检查矩阵输入
    weights = ahp.calculate_weight()  # 计算权重
    CR_values = ahp.consistency_check()  # 一致性检验

    # 检查所有目标层的所有准则层的一致性比率是否都小于 0.1
    all_pass = True
    for z, z_CR in enumerate(CR_values):
        for a, CR in enumerate(z_CR):
            if CR >= 0.1:
                print(f"目标层 Z{z+1} 和准则层 A{a+1} 的一致性比率 CR={CR} 大于等于 0.1，未通过一致性检验。")
                all_pass = False

    if all_pass == True:  # 如果所有的一致性比率都小于 0.1
        print("一致性检验通过")
        final_weights = ahp.calculate_final_weights(weights)  # 计算最终权重
        optimal_solution = ahp.find_optimal_solution(final_weights)  # 找到最优方案
        ahp.output_results(weights, CR_values, final_weights, optimal_solution)  # 输出结果
    else:
        print("一致性检验未通过，请重新输入矩阵。")


"""
算术平均法求权重

第一步:按列归一化,即An/Sum(A1+A2+...+An);
第二步:计算每行的算术平均值ω，即权重.


几何平均法

第一步:将A的元素按照行相乘得到新的一个列向量;
第二步:将新的向量的每个分量开n次方;
第三步:对该向量做归一化可得到权重.

"""