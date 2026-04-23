import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# dummy index
vx=0;vy=1;vz=2;pr=3;ro=4;bx=5;by=6;bz=7;ps=8

# Function to measure the reconnection rate
def measure_reconnection_rate(data_array, eta_1, num, x, y):
    num = num + 1
    y_center = y.size // 2
    point_5 = y_center + int(5/(y[1] - y[0]))
    time = np.linspace(0, 250, num)
    reconnection_rate = np.zeros(num)
    alfven_speed = np.zeros(num)
    B_outside = np.zeros(num)
    for number in range(num):
        data = data_array[number]
        data_pr = data[:, :, pr]
        data_bx = data[:, :, bx]
        data_by = data[:, :, by]
        data_ro = data[:, :, ro]
        B = np.sqrt(data_bx[0, point_5]**2 + data_by[0, point_5]**2)
        ro_arr = data_ro[0, point_5]
        alfven = np.sqrt(B**2 / ro_arr)
        jz_center = ((data_by[1, y_center] + data_by[1, y_center-1]) - (data_by[0, y_center] + data_by[0, y_center-1]))/(2 * (x[1]-x[0])) - ((data_bx[0, y_center] + data_bx[1, y_center]) - (data_bx[0, y_center-1] + data_bx[1, y_center-1]))/(2 * (y[y_center]-y[y_center-1]))

        reconnection_rate[number] = np.array(abs(eta_1 * jz_center / (alfven * B)))
        alfven_speed[number] = alfven
        B_outside[number] = B
    return reconnection_rate, alfven_speed, B_outside

import numpy as np
import openmhd

# dummy index
vx=0; vy=1; vz=2; pr=3; ro=4; bx=5; by=6; bz=7; ps=8


def read_and_mirror_dataset(
    data_dir_number,
    num=11,
    rank=16,
    xrange=(0.0, 130.0),
    yrange=(0.0, 15.0)
):
    """
    dataXX ディレクトリ (XX = data_dir_number) を読み込み、
    rank 分割データを結合 + y方向ミラーリングしたデータを返す。

    Returns
    -------
    data_all : ndarray (num, ix, jx, 9)
    x_all    : ndarray (ix,)
    y_all    : ndarray (jx,)
    """

    data_all = []
    t_all = np.zeros(num, dtype=np.double)

    for number in range(num):
        x_list = []
        data_list = []

        for i in range(rank):
            x, y, t, data = openmhd.data_read(
                f"./data/data{data_dir_number}/field-rank{i:05d}-{number:05d}.dat",
                i,
                xrange=xrange,
                yrange=yrange
            )

            # 領域重複の除去
            if i == 0:
                x_prev = x[:-1]
                data_prev = data[:-1, :, :]
            elif i == rank - 1:
                x_prev = x[1:]
                data_prev = data[1:, :, :]
            else:
                x_prev = x[1:-1]
                data_prev = data[1:-1, :, :]

            x_list.append(x_prev)
            data_list.append(data_prev)

        # x 方向結合
        x = np.concatenate(x_list, axis=0)
        data = np.concatenate(data_list, axis=0)

        # ---------- 2D mirroring (BC依存) ----------
        ix = x.size
        jx = 2 * y.size - 2
        jxh = y.size - 1

        tmp = data
        data = np.empty((ix, jx, 9), dtype=np.double)

        data[:, jxh:, :]   =  tmp[:, 1:, :]
        data[:, 0:jxh, :]  =  tmp[:, -1:-jxh-1:-1, :]

        data[:, 0:jxh, vy] = -tmp[:, -1:-jxh-1:-1, vy]
        data[:, 0:jxh, vz] = -tmp[:, -1:-jxh-1:-1, vz]
        data[:, 0:jxh, bx] = -tmp[:, -1:-jxh-1:-1, bx]
        data[:, 0:jxh, by] =  tmp[:, -1:-jxh-1:-1, by]
        data[:, 0:jxh, bz] =  tmp[:, -1:-jxh-1:-1, bz]
        # ------------------------------------------

        data_all.append(data)

        # 座標は最初の timestep のみ保存
        if number == 0:
            x_all = x
            tmpy = y
            y_all = np.empty(jx, dtype=np.double)
            y_all[jxh:]  =  tmpy[1:]
            y_all[0:jxh] = -tmpy[-1:-jxh-1:-1]

        # 時刻の保存
        t_all[number] = t
        

    return np.array(data_all), x_all, y_all, t_all

import numpy as np
import matplotlib.pyplot as plt

# dummy index
vx=0; vy=1; vz=2; pr=3; ro=4; bx=5; by=6; bz=7; ps=8


def plot_vx_with_blines(
    x, y, data, t,
    component=vx,
    cmap='jet',
    figsize=(10, 5),
    dpi=80,
    title_prefix='Outflow speed'
):
    """
    vx などのスカラー場 + 磁力線(Az等高線)を描画する

    Parameters
    ----------
    x, y : 1D ndarray
        座標
    data : ndarray (ix, iy, 9)
        MHDデータ
    t : float
        時刻
    component : int
        表示する成分 (vx, vy, ...)
    cmap : str
        カラーマップ
    """

    # ---------- figure ----------
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.clf()

    extent = [x[0], x[-1], y[0], y[-1]]

    # ---------- scalar field ----------
    tmp = np.empty((x.size, y.size), dtype=np.double)
    tmp[:, :] = data[:, :, component]

    mymax = max(tmp.max(), -tmp.min()) if tmp.max() > 0.0 else 0.0
    mymin = min(tmp.min(), -tmp.max()) if tmp.min() < 0.0 else 0.0

    if component == vx:
        mymax = 1.2
        mymin = -1.2
    elif component == ro:
        mymax = 12.0
        mymin = 0.0
    elif component == pr:
        mymax = 1.0
        mymin = 0.0

    img = plt.imshow(
        tmp.T,
        origin='lower',
        vmin=mymin,
        vmax=mymax,
        cmap=cmap,
        extent=extent,
        aspect='auto'
    )

    plt.colorbar(img)

    plt.xlabel("X", size=16)
    plt.ylabel("Y", size=16)
    plt.title(f'{title_prefix} (t = {t:6.1f})', size=20)

    # ---------- vector potential Az ----------
    az = np.empty((x.size, y.size), dtype=np.double)
    fx = 0.5 * (x[1] - x[0])
    fy = 0.5 * (y[1] - y[0])

    az[0, 0] = fy * data[0, 0, bx] - fx * data[0, 0, by]

    for j in range(1, y.size):
        az[0, j] = az[0, j-1] + fy * (data[0, j-1, bx] + data[0, j, bx])

    for i in range(1, x.size):
        az[i, :] = az[i-1, :] - fx * (data[i-1, :, by] + data[i, :, by])

    # ---------- magnetic field lines ----------
    plt.contour(
        az.T,
        extent=extent,
        colors='w',
        linestyles='solid'
    )

    plt.tight_layout()
    plt.show()

def plot_az_only(
    x, y, data, t,
    dA=0.02,
    A_min=None,
    A_max=None,
    cmap=None,
    interp=False,
    interp_factor=2,
    figsize=(6, 5),
    dpi=100,
    linewidth=0.7,
    color='k'
):

    extent = [x[0], x[-1], y[0], y[-1]]

    # ---------- compute Az ----------
    az = compute_az(x, y, data)

    # ---------- subtract offset (optional but recommended) ----------
    az = az - az.mean()

    # ---------- interpolation ----------
    if interp:
        from scipy.ndimage import zoom
        az_plot = zoom(az, interp_factor, order=3)
    else:
        az_plot = az

    # ---------- contour levels (FIXED spacing) ----------
    Amin = az_plot.min() if A_min is None else A_min
    Amax = az_plot.max() if A_max is None else A_max

    levels = np.arange(Amin, Amax, dA)

    # ---------- figure ----------
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.clf()

    if cmap is not None:
        img = plt.imshow(
            az_plot.T,
            origin='lower',
            extent=extent,
            cmap=cmap,
            aspect='auto'
        )
        plt.colorbar(img, label='Az')

    plt.contour(
        az_plot.T,
        extent=extent,
        levels=levels,
        colors=color,
        linewidths=linewidth
    )

    plt.xlabel("X", size=14)
    plt.ylabel("Y", size=14)
    plt.title(f"Magnetic field lines (ΔAz={dA:g})  t = {t:6.1f}", size=16)

    plt.tight_layout()
    plt.show()


def compute_az(x, y, data):
    az = np.empty((x.size, y.size))
    fx = 0.5 * (x[1] - x[0])
    fy = 0.5 * (y[1] - y[0])

    az[0, 0] = fy * data[0, 0, bx] - fx * data[0, 0, by]

    for j in range(1, y.size):
        az[0, j] = az[0, j-1] + fy * (data[0, j-1, bx] + data[0, j, bx])

    for i in range(1, x.size):
        az[i, :] = az[i-1, :] - fx * (data[i-1, :, by] + data[i, :, by])

    return az



def init_vx_with_blines(
    fig, ax, x, y, data0, component=vx,
    cmap='jet', title_prefix=''
):
    extent = [x[0], x[-1], y[0], y[-1]]

    tmp = data0[:, :, component]

    if component == vx:
        vmin, vmax = -1.2, 1.2
    elif component == ro:
        vmin, vmax = 0.0, 12.0
    elif component == pr:
        vmin, vmax = 0.0, 2.0
    else:
        vmin, vmax = tmp.min(), tmp.max()

    img = ax.imshow(
        tmp.T,
        origin='lower',
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect='auto'
    )

    cbar = fig.colorbar(img, ax=ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    title = ax.set_title("")

    # ---- Az 初期計算 ----
    az = compute_az(x, y, data0)
    cont = ax.contour(
        az.T,
        extent=extent,
        colors='w',
        linestyles='solid'
    )

    return img, cont, title


def plot_vx_and_compare(
    x, y, data, data2, t,
    component=vx,
    cmap='jet',
    figsize=(10, 5),
    dpi=80,
    title_prefix='Outflow speed'
):
    """
    vx などのスカラー場 + 磁力線(Az等高線)を描画する

    Parameters
    ----------
    x, y : 1D ndarray
        座標
    data : ndarray (ix, iy, 9)
        MHDデータ
    t : float
        時刻
    component : int
        表示する成分 (vx, vy, ...)
    cmap : str
        カラーマップ
    """

    # ---------- figure ----------
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.clf()

    extent = [x[0], x[-1], y[0], y[-1]]

    # ---------- scalar field ----------
    tmp = np.empty((x.size, y.size), dtype=np.double)
    tmp[:, :] = data[:, :, component]
    tmp2 = np.empty((x.size, y.size), dtype=np.double)
    tmp2[:, :] = data2[:, :, component]

    mymax = max(tmp.max(), -tmp.min()) if tmp.max() > 0.0 else 0.0
    mymin = min(tmp.min(), -tmp.max()) if tmp.min() < 0.0 else 0.0

    if component == vx:
        mymax = 1.2
        mymin = -1.2
    elif component == ro:
        mymax = 12.0
        mymin = 0.0
    elif component == pr:
        mymax = 2.0
        mymin = 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    img = ax1.imshow(
        tmp.T,
        origin='lower',
        vmin=mymin,
        vmax=mymax,
        cmap=cmap,
        extent=extent,
        aspect='auto'
    )

    img2 = ax2.imshow(
        tmp2.T,
        origin='lower',
        vmin=mymin,
        vmax=mymax,
        cmap=cmap,
        extent=extent,
        aspect='auto'
    )

    fig.colorbar(img, ax=ax1)
    fig.colorbar(img2, ax=ax2)
    ax1.set_xlabel("X", size=16)
    ax1.set_ylabel("Y", size=16)
    ax1.set_title(f'{title_prefix} (t = {t:6.1f})', size=20)
    ax2.set_xlabel("X", size=16)
    ax2.set_ylabel("Y", size=16)
    ax2.set_title(f'{title_prefix} (t = {t:6.1f})', size=20)

    # ---------- vector potential Az ----------
    az = np.empty((x.size, y.size), dtype=np.double)
    az2 = np.empty((x.size, y.size), dtype=np.double)
    fx = 0.5 * (x[1] - x[0])
    fy = 0.5 * (y[1] - y[0])

    az[0, 0] = fy * data[0, 0, bx] - fx * data[0, 0, by]
    az2[0, 0] = fy * data2[0, 0, bx] - fx * data2[0, 0, by]

    for j in range(1, y.size):
        az[0, j] = az[0, j-1] + fy * (data[0, j-1, bx] + data[0, j, bx])
        az2[0, j] = az2[0, j-1] + fy * (data2[0, j-1, bx] + data2[0, j, bx])

    for i in range(1, x.size):
        az[i, :] = az[i-1, :] - fx * (data[i-1, :, by] + data[i, :, by])
        az2[i, :] = az2[i-1, :] - fx * (data2[i-1, :, by] + data2[i, :, by])

    # ---------- magnetic field lines ----------
    ax1.contour(
        az.T,
        extent=extent,
        colors='w',
        linestyles='solid'
    )
    ax2.contour(
        az2.T,
        extent=extent,
        colors='w',
        linestyles='solid'
    )

    plt.tight_layout()
    plt.show()

def plot_something(
    x, y, data, t,
    cmap='jet',
    figsize=(10, 5),
    dpi=80,
    title_prefix='Outflow speed',
    logscale=False
):
    """
    vx などのスカラー場 + 磁力線(Az等高線)を描画する

    Parameters
    ----------
    x, y : 1D ndarray
        座標
    data : ndarray (ix, iy, 9)
        MHDデータ
    t : float
        時刻
    component : int
        表示する成分 (vx, vy, ...)
    cmap : str
        カラーマップ
    """

    # ---------- figure ----------
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.clf()

    extent = [x[0], x[-1], y[0], y[-1]]

    # ---------- scalar field ----------
    tmp = np.empty((x.size, y.size), dtype=np.double)
    tmp[:, :] = data

    if logscale:
        positive = tmp[tmp > 0]

        if positive.size == 0:
            # log 不可能 → 線形にフォールバック
            print("WARNING: no positive values, fallback to linear scale")
            logscale = False

    if logscale:
        vmin = 1e-1 #positive.min()
        vmax = positive.max()

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            print("WARNING: invalid vmin/vmax, fallback to linear scale")
            logscale = False


    if logscale:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        img = plt.imshow(
            tmp.T,
            origin='lower',
            cmap=cmap,
            extent=extent,
            aspect='auto',
            norm=norm
        )
    else:
        mymax = max(tmp.max(), -tmp.min()) if tmp.max() > 0.0 else 0.0
        mymin = min(tmp.min(), -tmp.max()) if tmp.min() < 0.0 else 0.0

        img = plt.imshow(
            tmp.T,
            origin='lower',
            cmap=cmap,
            extent=extent,
            aspect='auto',
            vmin=mymin,
            vmax=mymax
        )

    plt.colorbar(img)

    plt.xlabel("X", size=16)
    plt.ylabel("Y", size=16)
    plt.title(f'{title_prefix} (t = {t:6.1f})', size=20)

    plt.tight_layout()
    plt.show()

def kappa_plus_eta(data, x, y, kappa_0, center, constant=False):
    T0 = 0.1
    ro0 = 6
    ix0 = np.argmin(np.abs(x-center))
    eta_0 = 1/1000
    eta_1 = 1/60
    eta = np.zeros((y.size),np.double)
    kappa_AD = np.zeros((y.size),np.double)
    kappa_AD_B = np.zeros((y.size),np.double)

    for j in range(y.size):
        eta[j] = eta_0 + (eta_1 - eta_0) * (np.cosh(np.sqrt(x[ix0]**2 + y[j]**2))) ** (-2)
    if constant:
        kappa_AD = kappa_0
    else:
        kappa_AD = kappa_0 * ((T0*data[ix0,:,ro]/data[ix0,:,pr]) ** 0.5) * ((ro0/data[ix0,:,ro]) ** 2)

    kappa_AD_B = kappa_AD * (data[ix0,:,bx]**2 + data[ix0,:,by]**2)

    return eta, kappa_AD_B

def kappa_plus_eta_y(data, x, y, kappa_0, center_y, constant=False):
    T0 = 0.1
    ro0 = 6
    iy0 = np.argmin(np.abs(y-center_y))
    eta_0 = 1/1000
    eta_1 = 1/60
    eta = np.zeros((x.size),np.double)
    kappa_AD = np.zeros((x.size),np.double)
    kappa_AD_B = np.zeros((x.size),np.double)

    for i in range(x.size):
        eta[i] = eta_0 + (eta_1 - eta_0) * (np.cosh(np.sqrt(x[i]**2 + y[iy0]**2))) ** (-2)
    if constant:
        kappa_AD = kappa_0
    else:
        kappa_AD = kappa_0 * ((T0*data[:,iy0,ro]/data[:,iy0,pr]) ** 0.5) * ((ro0/data[:,iy0,ro]) ** 2)

    kappa_AD_B = kappa_AD * (data[:,iy0,bx]**2 + data[:,iy0,by]**2)

    return eta, kappa_AD_B

def kappa_plus_eta_all(data, x, y, kappa_0, constant=False):
    T0 = 0.1
    ro0 = 6
    eta_0 = 1/1000
    eta_1 = 1/60
    eta = np.zeros((x.size, y.size),np.double)
    kappa_AD = np.zeros((x.size, y.size),np.double)
    kappa_AD_B = np.zeros((x.size, y.size),np.double)

    for i in range(x.size):
        for j in range(y.size):
            eta[i, j] = eta_0 + (eta_1 - eta_0) * (np.cosh(np.sqrt(x[i]**2 + y[j]**2))) ** (-2)
    if constant:
        kappa_AD = kappa_0
    else:
        kappa_AD = kappa_0 * ((T0*data[:,:,ro]/data[:,:,pr]) ** 0.5) * ((ro0/data[:,:,ro]) ** 2)

    kappa_AD_B = kappa_AD * (data[:,:,bx]**2 + data[:,:,by]**2)

    return eta, kappa_AD_B