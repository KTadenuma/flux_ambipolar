import numpy as np
from scipy.stats import t
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t

def pca(data_section, i1, j1):
    
    X, Y = np.where(data_section >= 1)
    # print(X, Y)

    # 最小二乗法
    A = np.vstack([X, np.ones(len(X))]).T
    y_diff = np.max(Y) - np.min(Y)
    x_diff = X[np.argmax(Y)] - X[np.argmin(Y)]
    # print("傾きの目安:", y_diff , x_diff)
    m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
    #print("傾き:", m)
    #print("切片:", c)
    nxv = m/np.sqrt(m**2+1)
    nyv = -1/np.sqrt(m**2+1)
    #print("法線単位ベクトル", nxv, nyv)

    # A = np.vstack([X, np.ones(len(X))]).T
    # Y = np.array(Y)

    params, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
    if (len(residuals) == 0):
        residuals = [0]
    m, c = params

    # 残差平方和 → 分散推定
    n = len(Y)
    sigma2 = residuals[0] / (n - 2)  # 不偏分散

    # 共分散行列
    cov = sigma2 * np.linalg.inv(A.T @ A)
    stderr = np.sqrt(np.diag(cov))  # 標準誤差

    # 信頼区間（95%）
    alpha = 0.05
    tval = t.ppf(1 - alpha/2, n - 2)

    m_ci = (m - tval * stderr[0], m + tval * stderr[0])
    c_ci = (c - tval * stderr[1], c + tval * stderr[1])
    #print("切片のi, j", i1, j1 + c)
    #print("切片のi, jのCI", i1, j1 + c_ci[0], j1 + c_ci[1])

    # 法線の単位ベクトルの不確かさ
    nxv_ci = (m_ci[0]/np.sqrt(m_ci[0]**2+1), m_ci[1]/np.sqrt(m_ci[1]**2+1))
    nyv_ci = (-1/np.sqrt(m_ci[1]**2+1), -1/np.sqrt(m_ci[0]**2+1))
    #print("x成分の不確かさ CI:", nxv_ci)
    #print("y成分の不確かさ CI:", nyv_ci)

    #print("-----転置モード-----")
    # 転置モード
    A_t = np.vstack([Y, np.ones(len(Y))]).T
    params_t, residuals_t, rank_t, s_t = np.linalg.lstsq(A_t, X, rcond=None)
    if (len(residuals_t) == 0):
        residuals_t = [0]
    m_t, c_t = params_t
    #print("転置モード 傾き:", m_t)
    #print("転置モード 切片:", c_t)
    # 元の座標系での法線ベクトル
    nxv_t = 1/np.sqrt(m_t**2+1)
    nyv_t = -m_t/np.sqrt(m_t**2+1)
    #print("転置モード 法線単位ベクトル", nxv_t, nyv_t)
    # 残差平方和 → 分散推定
    sigma2_t = residuals_t[0] / (n - 2)  # 不偏分散
    # 共分散行列
    cov_t = sigma2_t * np.linalg.inv(A_t.T @ A_t)
    stderr_t = np.sqrt(np.diag(cov_t))  # 標準誤差
    # 信頼区間（95%）
    m_ci_t = (m_t - tval * stderr_t[0], m_t + tval * stderr_t[0])
    c_ci_t = (c_t - tval * stderr_t[1], c_t + tval * stderr_t[1])
    #print("転置モード 切片のi, j", i1 + c_t, j1)
    #print("転置モード 切片のi, jのCI", i1 + c_ci_t[0], i1 + c_ci_t[1], j1)
    if (abs(nxv) > 1/np.sqrt(2) or (x_diff+1e-8)/(y_diff+1e-8) < 1e-5):
        return nxv_t, nyv_t, m_t, c_t, (i1 + c_t, j1)
    else:
        return nxv, nyv, m, c, (i1, j1 + c)

def variable_graph(x, y, data_bx, data_by, nxv, nyv, i1, i2, j1, j2):
    bx = data_bx
    by = data_by
    bn = bx*nxv + by*nyv
    bt = -bx*nyv + by*nxv

    # --- 例として同じデータ構造 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 5))
    ax1, ax2, ax3, ax4 = axes.ravel()
    extent = [x[0], x[-1], y[0], y[-1]]

    # --- imshow 1 ---
    im1 = ax1.imshow(bn.T, origin='lower', extent=extent, aspect='auto')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax1)
    ax1.set_title('B_n')

    # --- imshow 2 ---
    im2 = ax2.imshow(bt.T, origin='lower', extent=extent, aspect='auto')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2)
    ax2.set_title('B_t')

    # --- line plots ---
    if (abs(nxv) <= 1/np.sqrt(2)):
        ax3.plot(y, bn[len(x)//2, :], label='B_n at mid x')
        ax3.set_title('B_n along mid x')
        ax3.grid()
        ax4.plot(y, bt[len(x)//2, :], label='B_t at mid x')
        ax4.set_title('B_t along mid x')
        ax4.grid()
    else:
        ax3.plot(x, bn[:, len(y)//2], label='B_n at mid y')
        ax3.set_title('B_n along mid y')
        ax3.grid()
        ax4.plot(x, bt[:, len(y)//2], label='B_t at mid y')
        ax4.set_title('B_t along mid y')
        ax4.grid()

    plt.tight_layout()
    plt.show()
    return bn, bt

def temperature_graph(x, y, data_pr, data_ro):
    pr = data_pr
    ro = data_ro
    tr = pr / ro  # 温度

    # --- 例として同じデータ構造 ---
    fig, ax = plt.subplots(figsize=(4, 2))
    extent = [x[0], x[-1], y[0], y[-1]]

    # --- imshow ---
    im = ax.imshow(tr.T, origin='lower', extent=extent, aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_title('Temperature (T = P / R)')

    plt.tight_layout()
    plt.show()
    return tr

def zero_and_one(x, y, data, m, c):
    extent = [x[0], x[-1], y[0], y[-1]]
    fig, ax = plt.subplots(figsize=(6, 5))
    # --- b の配列と同じ形状のインデックス配列を作成 ---
    ny, nx = data.shape  # 注意：bx.shape = (i方向, j方向)
    I, J = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    b = (J > m * I + c).astype(float)

    # --- 描画 ---
    im = ax.imshow(b.T, origin='lower', extent=extent, aspect='auto', cmap='gray', alpha=0.3)
    fig.colorbar(im, ax=ax)
    plt.show()
    return b

def zero_and_one_rev(x, y, data, m, c, err):
    extent = [x[0], x[-1], y[0], y[-1]]
    fig, ax = plt.subplots(figsize=(6, 5))
    # --- b の配列と同じ形状のインデックス配列を作成 ---
    ny, nx = data.shape  # 注意：bx.shape = (i方向, j方向)
    I, J = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    if (abs(m) < 1):
        b = (J > m * I + c - err).astype(float)
        c = - (J < m * I + c + err).astype(float)
    else:
        b = (I < (J - c)/m + err).astype(float)
        c = - (I > (J - c)/m - err).astype(float)
    b = b + c

    # --- 描画 ---
    im = ax.imshow(b.T, origin='lower', extent=extent, aspect='auto', cmap='gray', alpha=0.3)
    fig.colorbar(im, ax=ax)
    plt.show()
    return b

def zero_and_one_filter(x, y, data, m, c, err, err2):
    extent = [x[0], x[-1], y[0], y[-1]]
    fig, ax = plt.subplots(figsize=(4, 2))
    # --- b の配列と同じ形状のインデックス配列を作成 ---
    ny, nx = data.shape  # 注意：bx.shape = (i方向, j方向)
    I, J = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    if (abs(m) < 1):
        b = ((J > m * I + c - err) & (J < m * I + c + err2)).astype(float)
        c = - ((J < m * I + c + err) & (J > m * I + c - err2)).astype(float)
    else:
        b = ((I < (J - c)/m + err) & (I > (J - c)/m - err2)).astype(float)
        c = - ((I > (J - c)/m - err) & (I < (J - c)/m + err2)).astype(float)
    b = b + c

    # --- 描画 ---
    im = ax.imshow(b.T, origin='lower', extent=extent, aspect='auto', cmap='gray', alpha=0.3)
    fig.colorbar(im, ax=ax)
    plt.show()
    return b

def zero_and_one_temp(x, y, data, data_pr, data_ro, m, c, err, err2):

    pr = data_pr
    ro = data_ro
    tr = pr / ro  # 温度

    #extent = [x[0], x[-1], y[0], y[-1]]
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))
    # --- b の配列と同じ形状のインデックス配列を作成 ---
    ny, nx = data.shape  # 注意：bx.shape = (i方向, j方向)
    I, J = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    if (abs(m) < 1):
        b = ((J > m * I + c - err) & (J < m * I + c + err2)).astype(float)
        c = - ((J < m * I + c + err) & (J > m * I + c - err2)).astype(float)
    else:
        b = ((I < (J - c)/m + err) & (I > (J - c)/m - err2)).astype(float)
        c = - ((I > (J - c)/m - err) & (I < (J - c)/m + err2)).astype(float)
    b = b + c

    # --- 描画 ---
    #im1= ax1.imshow(b.T, origin='lower', extent=extent, aspect='auto', cmap='gray', alpha=0.3)
    #fig.colorbar(im1, ax=ax1)
    #im2= ax2.imshow(tr.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
    #fig.colorbar(im2, ax=ax2)
    #plt.show()
    return b, tr

# --- 信頼区間関数 ---
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)
    h = sem * t.ppf((1 + confidence) / 2., n - 1)
    return mean - h, mean + h

def review_variable(bn0, bn1, bt0, bt1, Mn0, Mn1, Mt0, Mt1):
    # --- 平均と信頼区間のまとめ ---
    results = {
    "Quantity": ["B_n", "B_n", "B_t", "B_t", "M_n", "M_n", "M_t", "M_t"],
    "b":        [0, 1, 0, 1, 0, 1, 0, 1],
    "Mean": [
        np.mean(bn0), np.mean(bn1),
        np.mean(bt0), np.mean(bt1),
        np.mean(Mn0), np.mean(Mn1),
        np.mean(Mt0), np.mean(Mt1)
    ],
    "95% CI (lower)": [
        confidence_interval(bn0)[0], confidence_interval(bn1)[0],
        confidence_interval(bt0)[0], confidence_interval(bt1)[0],
        confidence_interval(Mn0)[0], confidence_interval(Mn1)[0],
        confidence_interval(Mt0)[0], confidence_interval(Mt1)[0]
    ],
    "95% CI (upper)": [
        confidence_interval(bn0)[1], confidence_interval(bn1)[1],
        confidence_interval(bt0)[1], confidence_interval(bt1)[1],
        confidence_interval(Mn0)[1], confidence_interval(Mn1)[1],
        confidence_interval(Mt0)[1], confidence_interval(Mt1)[1]
        ]
    }

    df = pd.DataFrame(results)

    # 小数点の整形
    pd.set_option("display.precision", 6)
    print(df)

def review_variable_rev(bn0, bn1, bt0, bt1, Mn0, Mn1, Mt0, Mt1, T0, T1):
    # --- 平均と信頼区間のまとめ ---
    results = {
    "Quantity": ["B_n", "B_n", "B_t", "B_t", "M_n", "M_n", "M_t", "M_t", "T", "T"],
    "b":        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "Mean": [
        np.mean(bn0), np.mean(bn1),
        np.mean(bt0), np.mean(bt1),
        np.mean(Mn0), np.mean(Mn1),
        np.mean(Mt0), np.mean(Mt1),
        np.mean(T0), np.mean(T1)
    ],
    "95% CI (lower)": [
        confidence_interval(bn0)[0], confidence_interval(bn1)[0],
        confidence_interval(bt0)[0], confidence_interval(bt1)[0],
        confidence_interval(Mn0)[0], confidence_interval(Mn1)[0],
        confidence_interval(Mt0)[0], confidence_interval(Mt1)[0],
        confidence_interval(T0)[0], confidence_interval(T1)[0]
    ],
    "95% CI (upper)": [
        confidence_interval(bn0)[1], confidence_interval(bn1)[1],
        confidence_interval(bt0)[1], confidence_interval(bt1)[1],
        confidence_interval(Mn0)[1], confidence_interval(Mn1)[1],
        confidence_interval(Mt0)[1], confidence_interval(Mt1)[1],
        confidence_interval(T0)[1], confidence_interval(T1)[1]
        ]
    }

    if (np.mean(T1) > np.mean(T0)):
        print("0側が上流、1側が下流")
        if (np.mean(bt1) * np.mean(bt0) >= 0 and (abs(np.mean(bt1)) > abs(np.mean(bt0)))):
            print("ファストショック")
        elif (np.mean(bt1) * np.mean(bt0) >= 0 and (abs(np.mean(bt1)) <= abs(np.mean(bt0)))):
            print("スローショック")
        else:
            print("中間衝撃波")
    else:
        print("1側が上流、0側が下流")
        if (np.mean(bt0) * np.mean(bt1) >= 0 and (abs(np.mean(bt0)) > abs(np.mean(bt1)))):
            print("ファストショック")
        elif (np.mean(bt0) * np.mean(bt1) >= 0 and (abs(np.mean(bt0)) <= abs(np.mean(bt1)))):
            print("スローショック")
        else:
            print("中間衝撃波")

        

    df = pd.DataFrame(results)

    # 小数点の整形
    pd.set_option("display.precision", 6)
    print(df)

def review_variable_upstream(bn0, bn1, bt0, bt1, Mn0, Mn1, Mt0, Mt1, T0, T1):
    # --- 平均と信頼区間のまとめ ---
    results = {
    "Quantity": ["B_n", "B_n", "B_t", "B_t", "M_n", "M_n", "M_t", "M_t", "T", "T"],
    "b":        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "Mean": [
        np.mean(bn0), np.mean(bn1),
        np.mean(bt0), np.mean(bt1),
        np.mean(Mn0), np.mean(Mn1),
        np.mean(Mt0), np.mean(Mt1),
        np.mean(T0), np.mean(T1)
    ],
    "95% CI (lower)": [
        confidence_interval(bn0)[0], confidence_interval(bn1)[0],
        confidence_interval(bt0)[0], confidence_interval(bt1)[0],
        confidence_interval(Mn0)[0], confidence_interval(Mn1)[0],
        confidence_interval(Mt0)[0], confidence_interval(Mt1)[0],
        confidence_interval(T0)[0], confidence_interval(T1)[0]
    ],
    "95% CI (upper)": [
        confidence_interval(bn0)[1], confidence_interval(bn1)[1],
        confidence_interval(bt0)[1], confidence_interval(bt1)[1],
        confidence_interval(Mn0)[1], confidence_interval(Mn1)[1],
        confidence_interval(Mt0)[1], confidence_interval(Mt1)[1],
        confidence_interval(T0)[1], confidence_interval(T1)[1]
        ]
    }

    df = pd.DataFrame(results)

    # 小数点の整形
    pd.set_option("display.precision", 6)
    print(df)

    if (np.mean(T1) > np.mean(T0)):
        print("0側が上流、1側が下流")
        if (np.mean(bt1) * np.mean(bt0) >= 0 and (abs(np.mean(bt1)) > abs(np.mean(bt0)))):
            print("ファストショック")
        elif (np.mean(bt1) * np.mean(bt0) >= 0 and (abs(np.mean(bt1)) <= abs(np.mean(bt0)))):
            print("スローショック")
        else:
            print("中間衝撃波")
        return 0
    else:
        print("1側が上流、0側が下流")
        if (np.mean(bt0) * np.mean(bt1) >= 0 and (abs(np.mean(bt0)) > abs(np.mean(bt1)))):
            print("ファストショック")
        elif (np.mean(bt0) * np.mean(bt1) >= 0 and (abs(np.mean(bt0)) <= abs(np.mean(bt1)))):
            print("スローショック")
        else:
            print("中間衝撃波")
        return 1

## --- mva法 ---

def mva_func_2d(data_x, data_y, i1, i2, j1, j2):
    nx = i2-i1+1-4
    ny = j2-j1+1-4
    print("data shape MVA:", data_x[i1+2:i2-1,j1+2:j2-1].shape)
    mxx = np.sum((data_x*data_x)[i1+2:i2-1,j1+2:j2-1])/(nx*ny) - np.sum(data_x[i1+2:i2-1,j1+2:j2-1])*np.sum(data_x[i1+2:i2-1,j1+2:j2-1])/(nx*ny*nx*ny)
    mxy = np.sum((data_x*data_y)[i1+2:i2-1,j1+2:j2-1])/(nx*ny) - np.sum(data_x[i1+2:i2-1,j1+2:j2-1])*np.sum(data_y[i1+2:i2-1,j1+2:j2-1])/(nx*ny*nx*ny)
    myy = np.sum((data_y*data_y)[i1+2:i2-1,j1+2:j2-1])/(nx*ny) - np.sum(data_y[i1+2:i2-1,j1+2:j2-1])*np.sum(data_y[i1+2:i2-1,j1+2:j2-1])/(nx*ny*nx*ny)
    x1 = 0.5*(mxx+myy+np.sqrt((mxx-myy)**2+4*mxy**2))
    x2 = 0.5*(mxx+myy-np.sqrt((mxx-myy)**2+4*mxy**2))
    return x1, x2, mxx, mxy, myy

def mva_func_i(data_x, data_y, i1, i2, j):
    nx = i2-i1+1-4
    mxx = np.sum((data_x*data_x)[i1+2:i2-1,j])/(nx) - np.sum(data_x[i1+2:i2-1,j])*np.sum(data_x[i1+2:i2-1,j])/(nx*nx)
    mxy = np.sum((data_x*data_y)[i1+2:i2-1,j])/(nx) - np.sum(data_x[i1+2:i2-1,j])*np.sum(data_y[i1+2:i2-1,j])/(nx*nx)
    myy = np.sum((data_y*data_y)[i1+2:i2-1,j])/(nx) - np.sum(data_y[i1+2:i2-1,j])*np.sum(data_y[i1+2:i2-1,j])/(nx*nx)
    x1 = 0.5*(mxx+myy+np.sqrt((mxx-myy)**2+4*mxy**2))
    x2 = 0.5*(mxx+myy-np.sqrt((mxx-myy)**2+4*mxy**2))
    return x1, x2, mxx, mxy, myy

def mva_func_j(data_x, data_y, i, j1, j2):
    ny = j2-j1+1-4
    mxx = np.sum((data_x*data_x)[i,j1+2:j2-1])/(ny) - np.sum(data_x[i,j1+2:j2-1])*np.sum(data_x[i,j1+2:j2-1])/(ny*ny)
    mxy = np.sum((data_x*data_y)[i,j1+2:j2-1])/(ny) - np.sum(data_x[i,j1+2:j2-1])*np.sum(data_y[i,j1+2:j2-1])/(ny*ny)
    myy = np.sum((data_y*data_y)[i,j1+2:j2-1])/(ny) - np.sum(data_y[i,j1+2:j2-1])*np.sum(data_y[i,j1+2:j2-1])/(ny*ny)
    x1 = 0.5*(mxx+myy+np.sqrt((mxx-myy)**2+4*mxy**2))
    x2 = 0.5*(mxx+myy-np.sqrt((mxx-myy)**2+4*mxy**2))
    return x1, x2, mxx, mxy, myy

def m_x2n(mxx, mxy, myy, x1, x2):
    m1 = mxx ; m2 = mxy-x2
    # m1=mxy-x2 ; m2=myy
    nxm, nym = ( m2, -m1 ) / np.sqrt(m1**2+m2**2)
    return nxm, nym

class MVAOptimizerBase:
    def __init__(self, data_x, data_y, mva_func, d_min=3, d_max=8, metric="max/mean"):
        """
        metric: 評価指標
            - "max/mean" : max / mean  (デフォルト)
            - "max/median" : max / median
            - "zscore" : (max - mean) / std
        """
        self.data_x = data_x
        self.data_y = data_y
        self.mva_func = mva_func
        self.d_min = d_min
        self.d_max = d_max
        self.metric = metric

        self.box = None
        self.ratios = None
        self.best_d = None
        self.best_ratio = None
        self.optimal_index = None
        self.best_indices = []  # 各dの最適インデックスを記録

    def compute_ratio(self, start_index):
        """各dに対して指定指標のスコアと最適インデックスを記録"""
        self.ratios = []
        self.best_indices = []

        for d_idx in range(self.d_max - self.d_min + 1):
            arr = self.box[d_idx]
            if self.metric == "max/mean":
                score = np.max(arr) / np.mean(arr)
            elif self.metric == "max/median":
                score = np.max(arr) / np.median(arr)
            elif self.metric == "zscore":
                score = (np.max(arr) - np.mean(arr)) / (np.std(arr) + 1e-10)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            best_idx = np.argmax(arr) + start_index
            self.ratios.append(score)
            self.best_indices.append(best_idx)
            print(f"d = {d_idx + self.d_min}, {self.metric} = {score:.3f}, at index = {best_idx}")

        self.best_d_index = np.argmax(self.ratios)
        self.best_d = self.best_d_index + self.d_min
        self.best_ratio = self.ratios[self.best_d_index]



class MVAOptimizerI(MVAOptimizerBase):
    """固定j、iを変化させる最適化（現行コードに対応）"""
    def __init__(self, data_x, data_y, mva_func_2d, j_optimal, i_start, i_end, d_min=3, d_max=8, metric="max/mean"):
        super().__init__(data_x, data_y, mva_func_2d, d_min, d_max, metric) # 親クラスのインスタンスが作られ、その引数がセットされる
        self.j_optimal = j_optimal # 引数に書かれた数字
        self.i_start = i_start
        self.i_end = i_end

    def run(self):
        box = np.zeros((self.d_max - self.d_min + 1, self.i_end - self.i_start))
        for d in range(self.d_min, self.d_max + 1):
            for i in range(self.i_start, self.i_end):
                i1, i2 = i - d, i + d
                j1, j2 = self.j_optimal - d, self.j_optimal + d
                x1, x2, *_ = self.mva_func(self.data_x, self.data_y, i1, i2, j1, j2)
                box[d - self.d_min, i - self.i_start] = x1 / x2
        self.box = box
        self.compute_ratio(self.i_start) # 親クラスのメソッドを呼び出す
        self.optimal_index = np.argmax(self.box[self.best_d_index]) + self.i_start


class MVAOptimizerJ(MVAOptimizerBase):
    """固定i、jを変化させる最適化"""
    def __init__(self, data_x, data_y, mva_func_2d, i_optimal, j_start, j_end, d_min=3, d_max=8, metric="max/mean"):
        super().__init__(data_x, data_y, mva_func_2d, d_min, d_max, metric)
        self.i_optimal = i_optimal
        self.j_start = j_start
        self.j_end = j_end

    def run(self):
        box = np.zeros((self.d_max - self.d_min + 1, self.j_end - self.j_start))
        for d in range(self.d_min, self.d_max + 1):
            for j in range(self.j_start, self.j_end):
                i1, i2 = self.i_optimal - d, self.i_optimal + d
                j1, j2 = j - d, j + d
                x1, x2, *_ = self.mva_func(self.data_x, self.data_y, i1, i2, j1, j2)
                box[d - self.d_min, j - self.j_start] = x1 / x2
        self.box = box
        self.compute_ratio(self.j_start) 
        self.optimal_index = np.argmax(self.box[self.best_d_index]) + self.j_start

class MVAOptimizerI1D(MVAOptimizerBase):
    """j固定・i方向のスライスで最適化 (mva_func_i を使用)"""
    def __init__(self, data_x, data_y, mva_func_i, j_optimal, i_start, i_end, d_min=3, d_max=8, metric="max/mean"):
        super().__init__(data_x, data_y, mva_func_i, d_min, d_max, metric)
        self.j_optimal = j_optimal
        self.i_start = i_start
        self.i_end = i_end

    def run(self):
        box = np.zeros((self.d_max - self.d_min + 1, self.i_end - self.i_start))
        for d in range(self.d_min, self.d_max + 1):
            for i in range(self.i_start, self.i_end):
                i1, i2 = i - d, i + d
                x1, x2, *_ = self.mva_func(self.data_x, self.data_y, i1, i2, self.j_optimal)
                box[d - self.d_min, i - self.i_start] = x1 / x2
        self.box = box
        self.compute_ratio(self.i_start) 
        self.optimal_index = np.argmax(self.box[self.best_d_index]) + self.i_start
        self.i_optimal = self.optimal_index

class MVAOptimizerJ1D(MVAOptimizerBase):
    """i固定・j方向のスライスで最適化 (mva_func_j を使用)"""
    def __init__(self, data_x, data_y, mva_func_j, i_optimal, j_start, j_end, d_min=3, d_max=8, metric="max/mean"):
        super().__init__(data_x, data_y, mva_func_j, d_min, d_max, metric)
        self.i_optimal = i_optimal
        self.j_start = j_start
        self.j_end = j_end

    def run(self):
        box = np.zeros((self.d_max - self.d_min + 1, self.j_end - self.j_start))
        for d in range(self.d_min, self.d_max + 1):
            for j in range(self.j_start, self.j_end):
                j1, j2 = j - d, j + d
                x1, x2, *_ = self.mva_func(self.data_x, self.data_y, self.i_optimal, j1, j2)
                box[d - self.d_min, j - self.j_start] = x1 / x2
        self.box = box
        self.compute_ratio(self.j_start) 
        self.optimal_index = np.argmax(self.box[self.best_d_index]) + self.j_start
        self.j_optimal = self.optimal_index

def mva_similarity_2d(nxv, nyv, m, d_range, data_x, data_y, i_optimal, j_optimal, mva_func_2d, m_x2n):
    plot_arr = np.zeros((2*m, 2*m))
    prot_arr_2 = np.zeros((2*m, 2*m))

    for i_x in range(i_optimal-m, i_optimal+m):
        for j_y in range(j_optimal-m, j_optimal+m):
            x1_o, x2_o, mxx_o, mxy_o, myy_o = mva_func_2d(data_x, data_y, i_x-d_range, i_x+d_range, j_y-d_range, j_y+d_range)
            nxm_o, nym_o = m_x2n(mxx_o, mxy_o, myy_o, x1_o, x2_o)
            f = (nxm_o*nxv+nym_o*nyv)/np.sqrt((nxm_o**2+nym_o**2)*(nxv**2+nyv**2))
            plot_arr[i_x - (i_optimal-m), j_y - (j_optimal-m)] = abs(f)
            prot_arr_2[i_x - (i_optimal-m), j_y - (j_optimal-m)] = x1_o / x2_o

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(prot_arr_2.T, origin='lower', extent=(i_optimal-m, i_optimal+m, j_optimal-m, j_optimal+m), norm=LogNorm(vmin=np.nanmin(prot_arr_2[prot_arr_2 > 0])))
    fig.colorbar(im1, ax=ax1, label='Protection Ratio (x1/x2)')
    ax1.set_title('Protection Ratio around Optimal Point')
    ax1.set_xlabel('i index')
    ax1.set_ylabel('j index')
    ax1.scatter(i_optimal, j_optimal, color='red', label='Optimal Point')
    ax1.set_xticks(np.arange(i_optimal - m, i_optimal + m + 1, 4))
    ax1.legend()
    im2 = ax2.imshow(plot_arr.T, origin='lower', extent=(i_optimal-m, i_optimal+m, j_optimal-m, j_optimal+m))
    fig.colorbar(im2, ax=ax2, label='Similarity')
    ax2.set_title('Similarity around Optimal Point')
    ax2.set_xlabel('i index')
    ax2.set_ylabel('j index')
    ax2.scatter(i_optimal, j_optimal, color='red', label='Optimal Point')
    ax2.set_xticks(np.arange(i_optimal - m, i_optimal + m + 1, 4))
    ax2.legend()
    plt.show()

    # 最大値とその座標
    idx = np.array((i_optimal - m, j_optimal - m)) + np.array(np.unravel_index(np.argmax(plot_arr), plot_arr.shape))
    idx = tuple(idx)
    return np.max(plot_arr), idx

def mva_similarity_i(nxv, nyv, m, d_range, data_x, data_y, i_optimal, j_optimal, mva_func_i, m_x2n):
    plot_arr = np.zeros((2*m, 2*m))
    prot_arr_2 = np.zeros((2*m, 2*m))

    for i_x in range(i_optimal-m, i_optimal+m):
        for j_y in range(j_optimal-m, j_optimal+m):
            x1_o, x2_o, mxx_o, mxy_o, myy_o = mva_func_i(data_x, data_y, i_x-d_range, i_x+d_range, j_y)
            nxm_o, nym_o = m_x2n(mxx_o, mxy_o, myy_o, x1_o, x2_o)
            f = (nxm_o*nxv+nym_o*nyv)/np.sqrt((nxm_o**2+nym_o**2)*(nxv**2+nyv**2))
            plot_arr[i_x - (i_optimal-m), j_y - (j_optimal-m)] = abs(f)
            prot_arr_2[i_x - (i_optimal-m), j_y - (j_optimal-m)] = x1_o / x2_o


    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(prot_arr_2.T, origin='lower', extent=(i_optimal-m, i_optimal+m, j_optimal-m, j_optimal+m), norm=LogNorm(vmin=np.nanmin(prot_arr_2[prot_arr_2 > 0])))
    fig.colorbar(im1, ax=ax1, label='Protection Ratio (x1/x2)')
    ax1.set_title('Protection Ratio around Optimal Point')
    ax1.set_xlabel('i index')
    ax1.set_ylabel('j index')
    ax1.scatter(i_optimal, j_optimal, color='red', label='Optimal Point')
    ax1.set_xticks(np.arange(i_optimal - m, i_optimal + m + 1, 4))
    ax1.legend()
    im2 = ax2.imshow(plot_arr.T, origin='lower', extent=(i_optimal-m, i_optimal+m, j_optimal-m, j_optimal+m))
    fig.colorbar(im2, ax=ax2, label='Similarity')
    ax2.set_title('Similarity around Optimal Point')
    ax2.set_xlabel('i index')
    ax2.set_ylabel('j index')
    ax2.scatter(i_optimal, j_optimal, color='red', label='Optimal Point')
    ax2.set_xticks(np.arange(i_optimal - m, i_optimal + m + 1, 4))
    ax2.legend()
    plt.show()

    # 最大値とその座標
    idx = np.array((i_optimal - m, j_optimal - m)) + np.array(np.unravel_index(np.argmax(plot_arr), plot_arr.shape))
    idx = tuple(idx)
    return np.max(plot_arr), idx

def mva_similarity_j(nxv, nyv, m, d_range, data_x, data_y, i_optimal, j_optimal, mva_func_j, m_x2n):
    plot_arr = np.zeros((2*m, 2*m))
    prot_arr_2 = np.zeros((2*m, 2*m))

    for i_x in range(i_optimal-m, i_optimal+m):
        for j_y in range(j_optimal-m, j_optimal+m):
            x1_o, x2_o, mxx_o, mxy_o, myy_o = mva_func_j(data_x, data_y, i_x, j_y-d_range, j_y+d_range)
            nxm_o, nym_o = m_x2n(mxx_o, mxy_o, myy_o, x1_o, x2_o)
            f = (nxm_o*nxv+nym_o*nyv)/np.sqrt((nxm_o**2+nym_o**2)*(nxv**2+nyv**2))
            plot_arr[i_x - (i_optimal-m), j_y - (j_optimal-m)] = abs(f)
            prot_arr_2[i_x - (i_optimal-m), j_y - (j_optimal-m)] = x1_o / x2_o


    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(prot_arr_2.T, origin='lower', extent=(i_optimal-m, i_optimal+m, j_optimal-m, j_optimal+m), norm=LogNorm(vmin=np.nanmin(prot_arr_2[prot_arr_2 > 0])))
    fig.colorbar(im1, ax=ax1, label='Protection Ratio (x1/x2)')
    ax1.set_title('Protection Ratio around Optimal Point')
    ax1.set_xlabel('i index')
    ax1.set_ylabel('j index')
    ax1.scatter(i_optimal, j_optimal, color='red', label='Optimal Point')
    ax1.set_xticks(np.arange(i_optimal - m, i_optimal + m + 1, 4))
    ax1.legend()
    im2 = ax2.imshow(plot_arr.T, origin='lower', extent=(i_optimal-m, i_optimal+m, j_optimal-m, j_optimal+m))
    fig.colorbar(im2, ax=ax2, label='Similarity')
    ax2.set_title('Similarity around Optimal Point')
    ax2.set_xlabel('i index')
    ax2.set_ylabel('j index')
    ax2.scatter(i_optimal, j_optimal, color='red', label='Optimal Point')
    ax2.set_xticks(np.arange(i_optimal - m, i_optimal + m + 1, 4))
    ax2.legend()
    plt.show()

    # 最大値とその座標
    idx = np.array((i_optimal - m, j_optimal - m)) + np.array(np.unravel_index(np.argmax(plot_arr), plot_arr.shape))
    idx = tuple(idx)
    return np.max(plot_arr), idx

def rh_analysis(data_pr, data_ro, data_bn, data_bt, data_Mn, data_Mt, vs, b_rev):
    bn0 = np.mean(data_bn[b_rev==-1])
    bt0 = np.mean(data_bt[b_rev==-1])
    pr0 = np.mean(data_pr[b_rev==-1])
    beta0 = (2 * pr0) / (bn0**2 + bt0**2)
    print("Plasma beta upstream beta0:", f"{beta0:.6f}")
    bn1 = np.mean(data_bn[b_rev==1])
    bt1 = np.mean(data_bt[b_rev==1])
    pr1 = np.mean(data_pr[b_rev==1])
    beta1 = (2 * pr1) / (bn1**2 + bt1**2)
    print("Plasma beta downstream beta1:", f"{beta1:.6f}")
    
    ro0 = np.mean(data_ro[b_rev==-1])
    ro1 = np.mean(data_ro[b_rev==1])
    ca0 = np.sqrt((bn0**2) / ro0)
    print("Alfven speed ca upstream ca0:", f"{ca0:.6f} units")
    ca1 = np.sqrt((bn1**2) / ro1)
    print("Alfven speed ca downstream ca1:", f"{ca1:.6f} units")

    Mn0 = np.mean(data_Mn[b_rev==-1])
    Mn1 = np.mean(data_Mn[b_rev==1])
    vn0 = np.mean(Mn0 / ro0)
    vn1 = np.mean(Mn1 / ro1)
    alfven_mach_number_0 = (vn0 - vs) / ca0
    print("Alfven Mach number M_A0:", f"{alfven_mach_number_0:.6f}")
    alfven_mach_number_1 = (vn1 - vs) / ca1
    print("Alfven Mach number M_A1:", f"{alfven_mach_number_1:.6f}")
    temperature_0 = pr0 / ro0
    temperature_1 = pr1 / ro1

    temperature_ratio_0 = temperature_1 / temperature_0
    temperature_ratio_1 = temperature_0 / temperature_1
    print("T1/T0", temperature_ratio_0)
    print("T0/T1", temperature_ratio_1)

def rh_analysis_upstream(data_pr, data_ro, data_bx, data_by, data_vx, data_vy, nx, ny, b_rev):
    # --- 基本物理量 ---
    data_bn = data_bx * nx + data_by * ny
    data_bt = -data_bx * ny + data_by * nx
    data_Mn = data_ro * (data_vx * nx + data_vy * ny)
    data_Mt = data_ro * (-data_vx * ny + data_vy * nx)
    data_vn = data_vx * nx + data_vy * ny
    data_vt = -data_vx * ny + data_vy * nx

    # --- 平均値抽出 ---
    mean_dict = lambda arr: {k: np.mean(arr[b_rev == k]) for k in [-1, 1]}
    pr, ro = mean_dict(data_pr), mean_dict(data_ro)
    bn, bt = mean_dict(data_bn), mean_dict(data_bt)
    Mn, Mt = mean_dict(data_Mn), mean_dict(data_Mt)
    bx, by = mean_dict(data_bx), mean_dict(data_by)
    vx, vy = mean_dict(data_vx), mean_dict(data_vy)
    vn, vt = mean_dict(data_vn), mean_dict(data_vt)

    # --- 上流/下流判定 ---
    temp_ratio = (pr[1] / ro[1]) / (pr[-1] / ro[-1])
    upstream, downstream = (-1, 1) if temp_ratio >= 1 else (1, -1)

    # --- 各向きデータ ---
    def side(k):
        ca_bn = np.sqrt(bn[k]**2 / ro[k])
        ca_b = np.sqrt((bn[k]**2 + bt[k]**2) / ro[k])
        cs = np.sqrt(5/3 * pr[k] / ro[k])
        beta = 2 * pr[k] / (bn[k]**2 + bt[k]**2)
        return dict(vn=vn, vt=vt, ca_bn=ca_bn, ca_b=ca_b, cs=cs, beta=beta)

    up, down = side(upstream), side(downstream)

    # --- ショック速度・マッハ数など ---
    vs = (ro[upstream]*vn[upstream] - ro[downstream]*vn[downstream]) / (ro[upstream] - ro[downstream])
    # ホフマン・テラー基準系
    vh = vt[upstream] - (vn[upstream] - vs) * (bt[upstream] / bn[upstream])
    # Ma_up = (up["vn"] - vs) / up["ca_b"]
    # Ma_down = (down["vn"] - vs) / down["ca_b"]
    # アルヴェンマッハ数（全磁場速度版）
    # Ma_up = np.sqrt((vn[upstream] - vs)**2 + vt[upstream]**2) / up["ca_b"]
    # Ma_down = np.sqrt((vn[downstream] - vs)**2 + vt[downstream]**2) / down["ca_b"]
    # アルヴェンマッハ数（法線磁場速度版）
    Ma_up = abs(vn[upstream] - vs) / up["ca_bn"]
    Ma_down = abs(vn[downstream] - vs) / down["ca_bn"]
    # 通常マッハ数
    M_up = np.sqrt((vn[upstream] - vs)**2 + (vt[upstream] - vh)**2) / up["cs"]
    M_down = np.sqrt((vn[downstream] - vs)**2 + (vt[downstream] - vh)**2) / down["cs"]
    T_ratio = (pr[downstream]/ro[downstream]) / (pr[upstream]/ro[upstream])

    # --- 結果表示 --
    print("Upstream Analysis:")
    print(f"Plasma beta upstream: {up['beta']:.6f}")
    print(f"Plasma beta downstream: {down['beta']:.6f}")
    print(f"Alfven Mach number upstream: {Ma_up:.6f}")
    print(f"Alfven Mach number downstream: {Ma_down:.6f}")
    print(f"Sound speed upstream: {up['cs']:.6f}")
    print(f"Sound speed downstream: {down['cs']:.6f}")
    print(f"Mach number upstream: {M_up:.6f}")
    print(f"Mach number downstream: {M_down:.6f}")
    print(f"Temperature ratio (downstream/upstream): {T_ratio:.6f}")
    print(f"Shock angle of B (upstream): {np.degrees(np.arctan(bt[upstream]/bn[upstream])):.6f}°")
    print(f"Shock angle of B (downstream): {np.degrees(np.arctan(bt[downstream]/bn[downstream])):.6f}°")
    print(f"Shock speed vs: {vs:.6f}")

    # --- ランキン・ユゴニオ方程式 ---
    gamma = 5/3
    def RH_eqs(k, sign):
        bxk, byk = bx[k], by[k]
        rok, prk = ro[k], pr[k]
        B2 = bxk**2 + byk**2
        return {
            "eq1": rok * (vn[k] - vs)**2 + prk + 0.5 * (bt[k]**2 + bn[k]**2),
            "eq2": rok * (vn[k] - vs) * (vt[k]) - bt[k]*bn[k],
            "eq3": (vn[k] - vs)*bt[k] - (vt[k] - vh)*bn[k],
            "eq4": gamma/(gamma-1)*prk/rok + 0.5*((vn[k] - vs)**2 + (vt[k] - vh)**2) 
        }

    eq_up, eq_down = RH_eqs(upstream, -1), RH_eqs(downstream, 1)
    for i in range(1, 5):
        print(f"Rankine-Hugoniot Equation Ratio {i}: {eq_up[f'eq{i}']/eq_down[f'eq{i}']:.6f}")

    # --- 上下流可視化マップ ---
    up_down_data = np.zeros_like(b_rev)
    up_down_data[b_rev == upstream] = 1
    up_down_data[b_rev == downstream] = -1

    # --- ベクトル正規化 ---
    def norm_vec(x, y):
        n = np.sqrt(x**2 + y**2)
        return x/n, y/n, n

    bxu, byu, Bu = norm_vec(bx[upstream], by[upstream])
    bxd, byd, Bd = norm_vec(bx[downstream], by[downstream])
    Mxu, Myu, Mu = norm_vec(ro[upstream]*vx[upstream], ro[upstream]*vy[upstream])
    Mxd, Myd, Md = norm_vec(ro[downstream]*vx[downstream], ro[downstream]*vy[downstream])

    # --- 描画 ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                         gridspec_kw={'width_ratios': [1, 1, 1]},
                         subplot_kw={'box_aspect': 1})  # 各axを正方形に

    ax1, ax2, ax3 = axes

    # Bベクトル
    ax1.quiver(0, 0, bxu, byu, color='r', label='B (Upstream)', angles='xy', scale_units='xy', scale=1)
    ax1.quiver(0, 0, bxd, byd, color='b', label='B (Downstream)', angles='xy', scale_units='xy', scale=1)
    ax1.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5), aspect='equal', title='Normal and Tangential Vectors')
    ax1.grid(); ax1.legend()
    ax1.text(-1.4, -1.05, f"|B| Up={Bu:.3f}\n|B| Down={Bd:.3f}", fontsize=10)

    # Mベクトル
    ax2.quiver(0, 0, Mxu, Myu, color='r', label='M (Upstream)', angles='xy', scale_units='xy', scale=1)
    ax2.quiver(0, 0, Mxd, Myd, color='b', label='M (Downstream)', angles='xy', scale_units='xy', scale=1)
    ax2.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5), aspect='equal', title='Mass Flux Vectors')
    ax2.grid(); ax2.legend()
    ax2.text(-1.4, -1.05, f"|M| Up={Mu:.3f}\n|M| Down={Md:.3f}", fontsize=10)

    # 上下流領域マップ
    ax3.imshow(up_down_data.T, cmap='gray', origin='lower')
    ax3.set(xlim=(0, up_down_data.shape[0]), ylim=(0, up_down_data.shape[1]),
            aspect='auto', title='Upstream (white) Regions')

    plt.show()


def rh_analysis_simplified(data_pr, data_ro, data_bx, data_by, data_vx, data_vy, nx, ny, b_rev):
    # --- 基本物理量 ---
    data_bn = data_bx * nx + data_by * ny
    data_bt = -data_bx * ny + data_by * nx
    data_Mn = data_ro * (data_vx * nx + data_vy * ny)
    data_Mt = data_ro * (-data_vx * ny + data_vy * nx)
    data_vn = data_vx * nx + data_vy * ny
    data_vt = -data_vx * ny + data_vy * nx

    # --- 平均値抽出 ---
    mean_dict = lambda arr: {k: np.mean(arr[b_rev == k]) for k in [-1, 1]}
    pr, ro = mean_dict(data_pr), mean_dict(data_ro)
    bn, bt = mean_dict(data_bn), mean_dict(data_bt)
    Mn, Mt = mean_dict(data_Mn), mean_dict(data_Mt)
    bx, by = mean_dict(data_bx), mean_dict(data_by)
    vx, vy = mean_dict(data_vx), mean_dict(data_vy)
    vn, vt = mean_dict(data_vn), mean_dict(data_vt)

    # --- 上流/下流判定 ---
    temp_ratio = (pr[1] / ro[1]) / (pr[-1] / ro[-1])
    upstream, downstream = (-1, 1) if temp_ratio >= 1 else (1, -1)

    # --- 各向きデータ ---
    def side(k):
        bn_norm = bn[k]/np.sqrt(bx[k]**2 + by[k]**2)
        bt_norm = bt[k]/np.sqrt(bx[k]**2 + by[k]**2)
        ca_bn = np.sqrt(bn[k]**2 / ro[k])
        ca_b = np.sqrt((bn[k]**2 + bt[k]**2) / ro[k])
        cs = np.sqrt(5/3 * pr[k] / ro[k])
        beta = 2 * pr[k] / (bn[k]**2 + bt[k]**2)
        return dict(vn=vn, vt=vt, bn_norm=bn_norm, bt_norm=bt_norm, ca_bn=ca_bn, ca_b=ca_b, cs=cs, beta=beta)

    up, down = side(upstream), side(downstream)

    # --- ショック速度・マッハ数など ---
    vs = (ro[upstream]*vn[upstream] - ro[downstream]*vn[downstream]) / (ro[upstream] - ro[downstream])
    # ホフマン・テラー基準系
    vh = vt[upstream] - (vn[upstream] - vs) * (bt[upstream] / bn[upstream])
    # アルヴェンマッハ数（法線磁場速度版）
    Ma_up = abs(vn[upstream] - vs) / up["ca_bn"]
    Ma_down = abs(vn[downstream] - vs) / down["ca_bn"]
    # アルヴェンマッハ数（完全版）
    F_Ma_up = np.sqrt((vn[upstream] - vs)**2 + (vt[upstream] - vh)**2) / up["ca_b"]
    F_Ma_down = np.sqrt((vn[downstream] - vs)**2 + (vt[downstream] - vh)**2) / down["ca_b"]
    # 通常マッハ数（法線速度版）
    M_up = abs(vn[upstream] - vs) / up["cs"]
    M_down = abs(vn[downstream] - vs) / down["cs"]
    # 通常マッハ数（完全版）
    F_M_up = np.sqrt((vn[upstream] - vs)**2 + (vt[upstream] - vh)**2) / up["cs"]
    F_M_down = np.sqrt((vn[downstream] - vs)**2 + (vt[downstream] - vh)**2) / down["cs"]
    T_ratio = (pr[downstream]/ro[downstream]) / (pr[upstream]/ro[upstream])

    shock_angle_up = np.degrees(np.arctan(bt[upstream]/bn[upstream]))
    shock_angle_down = np.degrees(np.arctan(bt[downstream]/bn[downstream]))

    kinetic_energy_up = 0.5 * ro[upstream] * ( (vn[upstream]-vs)**2 + (vt[upstream]-vh)**2 )
    magnetic_energy_up = 0.5 * (bx[upstream]**2 + by[upstream]**2)
    thermal_energy_up = pr[upstream] / (5/3 - 1)

    kinetic_energy_down = 0.5 * ro[downstream] * ( (vn[downstream]-vs)**2 + (vt[downstream]-vh)**2 )
    magnetic_energy_down = 0.5 * (bx[downstream]**2 + by[downstream]**2)
    thermal_energy_down = pr[downstream] / (5/3 - 1)

    # --- ランキン・ユゴニオ方程式 ---
    gamma = 5/3
    def RH_eqs(k, sign):
        bxk, byk = bx[k], by[k]
        rok, prk = ro[k], pr[k]
        B2 = bxk**2 + byk**2
        # print(- bt[k]*bn[k], rok * (vn[k] - vs) * (vt[k] - vh))
        return {
            "eq1": rok * (vn[k] - vs)**2 + prk + 0.5 * (bt[k]**2 + bn[k]**2),
            "eq2": rok * (vn[k] - vs) * (vt[k] - vh) - bt[k]*bn[k],
            "eq3": (vn[k] - vs)*bt[k] - (vt[k] - vh)*bn[k],
            "eq4": gamma/(gamma-1)*prk/rok + 0.5*((vn[k] - vs)**2 + (vt[k] - vh)**2),
            "eq5": abs((rok * (vn[k] - vs) * (vt[k] - vh)) / (bt[k]*bn[k]))
        }

    eq_up, eq_down = RH_eqs(upstream, -1), RH_eqs(downstream, 1)

    # --- 上下流可視化マップ ---
    up_down_data = np.zeros_like(b_rev)
    up_down_data[b_rev == upstream] = 1
    up_down_data[b_rev == downstream] = -1

    return {
        "upstream": up,
        "downstream": down,
        "vs": vs,
        "Ma_up": Ma_up,
        "Ma_down": Ma_down,
        "F_Ma_up": F_Ma_up,
        "F_Ma_down": F_Ma_down,
        "M_up": M_up,
        "M_down": M_down,
        "F_M_up": F_M_up,
        "F_M_down": F_M_down,
        "T_ratio": T_ratio,
        "shock_angle_up": shock_angle_up,
        "shock_angle_down": shock_angle_down,
        "RH_eqs_up": eq_up,
        "RH_eqs_down": eq_down,
        "up_down_map": up_down_data,
        "kinetic_energy_up": kinetic_energy_up,
        "magnetic_energy_up": magnetic_energy_up,
        "thermal_energy_up": thermal_energy_up,
        "kinetic_energy_down": kinetic_energy_down,
        "magnetic_energy_down": magnetic_energy_down,
        "thermal_energy_down": thermal_energy_down
    }

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
from io import BytesIO
from PIL import Image

# ==============================
# 表（df_analysis）を画像に変換
# ==============================
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def dataframe_to_image(df):
    """Pandas DataFrame を PNG 画像として返す"""
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(df) + 1))
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.3)

    buffer = BytesIO()
    fig.savefig(buffer, dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer)


def dataframe_to_image_two_rows(df):
    """横長 DataFrame を 2 段に分けて 1 枚の画像にまとめる"""

    # 列を半分に分割
    mid = len(df.columns) // 2
    df_top = df[df.columns[:mid]]
    df_bottom = df[df.columns[mid:]]

    # 各段を画像化
    img_top = dataframe_to_image(df_top)
    img_bottom = dataframe_to_image(df_bottom)

    # 横幅を揃える（必要）
    width = max(img_top.width, img_bottom.width)
    new_top = Image.new("RGB", (width, img_top.height), "white")
    new_bottom = Image.new("RGB", (width, img_bottom.height), "white")

    new_top.paste(img_top, (0, 0))
    new_bottom.paste(img_bottom, (0, 0))

    # 縦に連結
    total_height = new_top.height + new_bottom.height
    combined = Image.new("RGB", (width, total_height), "white")
    combined.paste(new_top, (0, 0))
    combined.paste(new_bottom, (0, new_top.height))

    return combined


# ==============================
# 2Dデータ上に矩形を描画
# ==============================
def plot_region_with_table(data_2, i1, i2, j1, j2, df_analysis):
    table_img = dataframe_to_image(df_analysis)

    fig, axes = plt.subplots(2, 1, figsize=(10, 20))

    # --- 1️⃣ 調査領域図 ---
    ax = axes[0]
    ax.imshow(data_2.T, origin='lower', cmap='gray')
    rect = Rectangle((i1, j1), i2 - i1, j2 - j1,
                     linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    ax.set_title("調査した領域 (赤枠)")
    ax.set_xlabel("i index")
    ax.set_ylabel("j index")

    # --- 2️⃣ 解析結果表（画像） ---
    axes[1].imshow(table_img)
    axes[1].axis('off')
    axes[1].set_title("解析結果テーブル (df_analysis)")

    plt.tight_layout()
    plt.show()


def rh_analysis_line(data_pr, data_ro, data_bx, data_by, data_vx, data_vy, nx, ny, b_rev):
    # --- 基本物理量 ---
    data_bn = data_bx * nx + data_by * ny
    data_bt = -data_bx * ny + data_by * nx
    data_Mn = data_ro * (data_vx * nx + data_vy * ny)
    data_Mt = data_ro * (-data_vx * ny + data_vy * nx)
    data_vn = data_vx * nx + data_vy * ny
    data_vt = -data_vx * ny + data_vy * nx

    # --- 平均値抽出 ---
    mean_dict = lambda arr: {k: np.mean(arr[b_rev == k]) for k in [-1, 1]}
    pr, ro = mean_dict(data_pr), mean_dict(data_ro)
    bn, bt = mean_dict(data_bn), mean_dict(data_bt)
    Mn, Mt = mean_dict(data_Mn), mean_dict(data_Mt)
    bx, by = mean_dict(data_bx), mean_dict(data_by)
    vx, vy = mean_dict(data_vx), mean_dict(data_vy)
    vn, vt = mean_dict(data_vn), mean_dict(data_vt)

    # --- 上流/下流判定 ---
    temp_ratio = (pr[1] / ro[1]) / (pr[-1] / ro[-1])
    upstream, downstream = (-1, 1) if temp_ratio >= 1 else (1, -1)

    # --- ショック速度・マッハ数など ---
    vs = (ro[upstream]*vn[upstream] - ro[downstream]*vn[downstream]) / (ro[upstream] - ro[downstream])
    vsx = vs * nx
    vsy = vs * ny
    # ホフマン・テラー基準系
    vh = vt[upstream] - (vn[upstream] - vs) * (bt[upstream] / bn[upstream])
    vhx = vh * -ny
    vhy = vh * nx

    # --- 上下流可視化マップ ---
    up_down_data = np.zeros_like(b_rev)
    up_down_data[b_rev == upstream] = 1
    up_down_data[b_rev == downstream] = -1

    # --- ベクトル正規化 ---
    def norm_vec(x, y):
        n = np.sqrt(x**2 + y**2)
        return x/n, y/n, n

    bxu, byu, Bu = norm_vec(bx[upstream], by[upstream])
    bxd, byd, Bd = norm_vec(bx[downstream], by[downstream])
    # ショック速度、ホフマン・テラー速度ベクトルも考慮
    Mxu, Myu, Mu = norm_vec(ro[upstream]*(vx[upstream] - vsx - vhx), ro[upstream]*(vy[upstream] - vsy - vhy))
    Mxd, Myd, Md = norm_vec(ro[downstream]*(vx[downstream] - vsx - vhx), ro[downstream]*(vy[downstream] - vsy - vhy))

    # --- 描画 ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                         gridspec_kw={'width_ratios': [1, 1, 1]},
                         subplot_kw={'box_aspect': 1})  # 各axを正方形に

    ax1, ax2, ax3 = axes

    # Bベクトル
    ax1.quiver(0, 0, bxu, byu, color='r', label='B (Upstream)', angles='xy', scale_units='xy', scale=1)
    ax1.quiver(0, 0, bxd, byd, color='b', label='B (Downstream)', angles='xy', scale_units='xy', scale=1)
    ax1.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5), aspect='equal', title='Normal and Tangential Vectors')
    ax1.grid(); ax1.legend()
    ax1.text(-1.4, -1.05, f"|B| Up={Bu:.3f}\n|B| Down={Bd:.3f}", fontsize=10)

    # 接線を引く（傾き = -nx/ny）
    if ny != 0:
        x_vals = np.array([-1.5, 1.5])
        y_vals = - (nx / ny) * x_vals
        ax1.plot(x_vals, y_vals, 'k--', label='Tangential Line')
        ax1.legend()
    else:
        ax1.axvline(0, color='k', linestyle='--', label='Tangential Line')
        ax1.legend()
    

    # Mベクトル
    ax2.quiver(0, 0, Mxu, Myu, color='r', label='M (Upstream)', angles='xy', scale_units='xy', scale=1)
    ax2.quiver(0, 0, Mxd, Myd, color='b', label='M (Downstream)', angles='xy', scale_units='xy', scale=1)
    ax2.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5), aspect='equal', title='Mass Flux Vectors')
    ax2.grid(); ax2.legend()
    ax2.text(-1.4, -1.05, f"|M| Up={Mu:.3f}\n|M| Down={Md:.3f}", fontsize=10)

    # 同様の接線を引く
    if ny != 0:
        ax2.plot(x_vals, y_vals, 'k--', label='Tangential Line')
        ax2.legend()
    else:
        ax2.axvline(0, color='k', linestyle='--', label='Tangential Line')
        ax2.legend()

    # 上下流領域マップ
    ax3.imshow(up_down_data.T, cmap='gray', origin='lower')
    ax3.set(xlim=(0, up_down_data.shape[0]), ylim=(0, up_down_data.shape[1]),
            aspect='auto', title='Upstream (white) Regions')

    plt.show()