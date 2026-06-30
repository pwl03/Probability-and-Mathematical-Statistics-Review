#!/usr/bin/env python3
"""
分位数计算工具
输入分布和分位数，输出对应的临界值
"""

import sys

def main():
    # 尝试导入必要的库
    try:
        from scipy import stats
        has_scipy = True
    except ImportError:
        has_scipy = False
        try:
            import numpy as np
            import math
            has_numpy = True
        except ImportError:
            has_numpy = False
            print("错误: 需要安装 scipy 或 numpy")
            print("请运行: pip install scipy")
            sys.exit(1)

    # 如果没有 scipy，提供一个纯 Python 的 erfinv 近似实现
    if not has_scipy:
        # 尝试从 numpy 导入 erfinv（numpy >= 2.0 支持）
        try:
            from numpy import sqrt, erfinv
        except ImportError:
            # numpy < 2.0 没有 erfinv，使用近似公式
            import math
            def erfinv(x):
                """逆误差函数的近似（Winitzki 近似）"""
                if abs(x) >= 1:
                    return math.copysign(1e10, x)
                a = 8 * (math.pi - 3) / (3 * math.pi * (4 - math.pi))
                s = math.copysign(1, x)
                x = abs(x)
                t = 2 / (math.pi * a) + math.log(1 - x * x) / 2
                return s * math.sqrt(-t + math.sqrt(t * t - math.log(1 - x * x) / a))
            sqrt = math.sqrt

    print("\n" + "="*50)
    print("分位数计算工具")
    print("="*50)
    print("分布简写:")
    print("  N   - 标准正态分布 (用法: N 0.975)")
    print("  t   - t分布        (用法: t 10 0.95)")
    print("  F   - F分布        (用法: F 5 10 0.95)")
    print("  X2  - 卡方分布     (用法: X2 8 0.95)")
    print("输入 'exit' 或 'quit' 退出")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input(">>> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("再见！")
                break

            parts = user_input.split()
            dist = parts[0].upper()

            # 根据分布解析参数
            if dist == 'N':
                if len(parts) != 2:
                    print("错误: N 格式应为: N [分位数]")
                    continue
                q = float(parts[1])
                params = []

            elif dist == 'T':
                if len(parts) != 3:
                    print("错误: t 格式应为: t [自由度] [分位数]")
                    continue
                df = float(parts[1])
                q = float(parts[2])
                params = [df]
                if df < 1:
                    print("错误: 自由度必须 >= 1")
                    continue

            elif dist == 'F':
                if len(parts) != 4:
                    print("错误: F 格式应为: F [dfn] [dfd] [分位数]")
                    continue
                dfn = float(parts[1])
                dfd = float(parts[2])
                q = float(parts[3])
                params = [dfn, dfd]
                if dfn < 1 or dfd < 1:
                    print("错误: 自由度必须 >= 1")
                    continue

            elif dist == 'X2':
                if len(parts) != 3:
                    print("错误: X2 格式应为: X2 [自由度] [分位数]")
                    continue
                df = float(parts[1])
                q = float(parts[2])
                params = [df]
                if df < 1:
                    print("错误: 自由度必须 >= 1")
                    continue

            else:
                print(f"错误: 未知分布 '{dist}'，支持的分布: N, t, F, X2")
                continue

            # 检查分位数范围
            if not (0 < q < 1):
                print("错误: 分位数必须在 (0,1) 区间内")
                continue

            # 如果没有 scipy，只有 N 和 t 可以用近似替代
            if not has_scipy:
                print("警告: scipy 未安装，部分功能受限")
                if dist == 'N':
                    value = sqrt(2) * erfinv(2*q - 1)
                    print(f"分位数: {value:.6f}")
                elif dist == 'T':
                    # t 分布的近似（基于正态近似的修正）
                    print("警告: 未安装 scipy，t 分布结果为近似值")
                    z = sqrt(2) * erfinv(2*q - 1)
                    value = z * (1 + (z*z + 1) / (4*df))
                    print(f"分位数(近似): {value:.6f}")
                else:
                    print(f"错误: {dist} 分布需要 scipy，请运行: pip install scipy")
                    continue
            else:
                # 使用 scipy 精确计算
                if dist == 'N':
                    value = stats.norm.ppf(q)
                elif dist == 'T':
                    value = stats.t.ppf(q, df=df)
                elif dist == 'F':
                    value = stats.f.ppf(q, dfn=dfn, dfd=dfd)
                elif dist == 'X2':
                    value = stats.chi2.ppf(q, df=df)
                else:
                    continue
                print(f"分位数: {value:.6f}")

        except ValueError as e:
            print(f"输入错误: 请确保输入的是数字。{e}")
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"计算错误: {e}")

if __name__ == "__main__":
    main()