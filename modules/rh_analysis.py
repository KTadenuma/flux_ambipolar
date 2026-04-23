import numpy as np
import matplotlib.pyplot as plt

def man_upstream(theta_1, beta_1, man_downstream, gamma=5/3): # theta_1 in degrees
    gamma_1 = (gamma-1)/gamma
    gamma_2 = (gamma+1)/(gamma-1)
    theta_1 = np.radians(theta_1)
    f1 = man_downstream**2 * (gamma_1 * (gamma_2 - np.tan(theta_1)**2) * (man_downstream**2 - 1)**2)
    f1 = f1 + man_downstream**2 * ((np.tan(theta_1)**2) * (gamma_1 * man_downstream**2 -1) * (man_downstream**2 -2))
    f1 = f1 - (beta_1/(np.cos(theta_1)**2)) * ((man_downstream**2 - 1)**2)

    f2 = (gamma_1/(np.cos(theta_1)**2)) * (man_downstream**2 - 1)**2
    f2 = f2 - man_downstream**2 * np.tan(theta_1)**2 * (gamma_1 * man_downstream**2 -1)

    if f1/f2 < 0:
        # print(f'Warning: No real solution for the upstream Mach number.{f1:.3f} / {f2:.3f} < 0')
        return np.nan
    else:
        man_upstream_value = np.sqrt(f1/f2)
        return man_upstream_value
    
def rh_analysis_graph(beta_up, theta_up, point, n=2):
    # rh_analysis_graph(beta_up, theta_up, point, point2, n=2):
    man_down = np.linspace(0.001, 20*point[0], 50000)
    man_up = np.zeros_like(man_down)
    man_up = np.vectorize(man_upstream)(theta_up, beta_up, man_down, gamma=5/3)
    tmp = man_up.copy()
    man_up = man_up[ np.isfinite(tmp) ]
    man_down = man_down[ np.isfinite(tmp) ]

    ## 描画用配列
    man_down_1 = np.linspace(0.001, n*point[0], 100)
    man_up_1 = np.zeros_like(man_down_1)
    man_up_1 = np.vectorize(man_upstream)(theta_up, beta_up, man_down_1, gamma=5/3)
    tmp = man_up_1.copy()
    man_up_1 = man_up_1[ np.isfinite(tmp) ]
    man_down_1 = man_down_1[ np.isfinite(tmp) ]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 10))

    # 元のプロット
    ax1.plot(man_down**2, man_up**2, label=f'θ={theta_up}°, β={beta_up}', linewidth=0.8)

    # 対角線 y = x
    vals = np.concatenate([man_up_1**2, man_down_1**2])

    if vals.size == 0:
        xmax = 1
    else:
        xmax = np.max(vals) + 0.2

    x = np.linspace(0, xmax, 200)
    ax1.plot(x, x, color='black', linewidth=0.3)
    ax1.plot(point[0]**2, point[1]**2, color='black', marker='o', markersize=3.8, label='Point 1')

    # y < x の領域を灰色に塗る
    ax1.fill_between(x, 0, x, color='gray', alpha=0.3)

    ax1.set_xlabel('(Downstream Mach Number)^2')
    ax1.set_ylabel('(Upstream Mach Number)^2')
    ax1.set_xlim(0, np.max(man_down_1**2) + 0.2)
    ax1.set_ylim(0, np.max(man_up_1**2) + 0.2)
    #ax1.set_xlim(0, 3)
    #ax1.set_ylim(0, 3)
    ax1.axvline(x=1.0, color='black', linestyle='-', linewidth=1.0)
    ax1.axhline(y=1.0, color='black', linestyle='-', linewidth=1.0)
    ax1.set_title('Upstream vs Downstream Mach Number')
    ax1.legend()
    ax1.grid()

    Bt_ratio = (man_up**2 -1)/(man_down**2 -1)
    rho_ratio = (man_up**2)/(man_down**2)
    p_ratio = 1 + 2 * man_up**2 * np.cos(np.radians(theta_up))**2 / beta_up * (1 - 1/rho_ratio) + np.tan(np.radians(theta_up))**2 * np.cos(np.radians(theta_up))**2 / beta_up * (1-Bt_ratio**2)
    p_ratio_point = 1 + 2 * point[1]**2 * np.cos(np.radians(theta_up))**2 / beta_up * (1 - 1/(point[1]**2/point[0]**2)) + np.tan(np.radians(theta_up))**2 * np.cos(np.radians(theta_up))**2 / beta_up * (1-((point[1]**2 -1)/(point[0]**2 -1))**2)
    s_ratio = p_ratio / rho_ratio**(5/3)
    s_ratio_point = p_ratio_point / (point[1]**2/point[0]**2)**(5/3)
    gamma = 5/3
    m_ratio = rho_ratio / p_ratio + gamma / (gamma -1) * beta_up / man_up**2 * (rho_ratio/p_ratio - 1)
    # m_ratio_point = (point[1]**2/point[0]**2) / p_ratio_point + gamma / (gamma -1) * beta_up / point[1]**2 * (1 - (point[1]**2/point[0]**2)/p_ratio_point)

    # 描画用配列

    Bt_ratio_1 = (man_up_1**2 -1)/(man_down_1**2 -1)
    rho_ratio_1 = (man_up_1**2)/(man_down_1**2)
    p_ratio_1 = 1 + 2 * man_up_1**2 * np.cos(np.radians(theta_up))**2 / beta_up * (1 - 1/rho_ratio_1) + np.tan(np.radians(theta_up))**2 * np.cos(np.radians(theta_up))**2 / beta_up * (1-Bt_ratio_1**2)
    s_ratio_1 = p_ratio_1 / rho_ratio_1**(5/3)


    ax2.plot(man_up**2, - Bt_ratio, label=f'θ={theta_up}°, β={beta_up}', linewidth=0.8)
    ax2.plot(point[1]**2, - (point[1]**2 -1)/(point[0]**2 -1), color='black', marker='o', markersize=3.8, label='Point 1')
    ax2.set_xlabel('(Upstream Mach Number)^2')
    ax2.set_ylabel('- Bt2 / Bt1')
    ax2.set_xlim(0, np.max(man_up_1**2) + 0.2)
    ax2.set_ylim(np.min(-Bt_ratio_1) - 0.2, np.max(-Bt_ratio_1) + 0.2)
    ax2.set_title('Ratio of Tangential Magnetic Fields')
    ax2.legend()
    ax2.grid()

    ax3.plot(man_up**2, rho_ratio, label=f'θ={theta_up}°, β={beta_up}', linewidth=0.8)
    ax3.plot(point[1]**2, (point[1]**2)/(point[0]**2), color='black', marker='o', markersize=3.8, label='Point 1')
    ax3.set_xlabel('(Upstream Mach Number)^2')
    ax3.set_ylabel('Density Ratio ρ2 / ρ1')
    ax3.set_title('Density Ratio Across the Shock')
    ax3.set_xlim(0, np.max(man_up_1**2) + 0.2)
    ax3.set_ylim(0, np.max(rho_ratio_1) + 0.2)
    ax3.legend()
    ax3.grid()

    ax3.fill_between(x, ax1.get_ylim()[0], 1, color='gray', alpha=0.3)


    # エントロピー比
    ax4.plot(man_up**2, s_ratio, label=f'θ={theta_up}°, β={beta_up}', linewidth=0.8)
    ax4.plot(point[1]**2, s_ratio_point, color='black', marker='o', markersize=3.8, label='Point 1')
    ax4.set_xlabel('(Upstream Mach Number)^2')
    ax4.set_ylabel('Entropy Ratio S2 / S1')
    ax4.set_title('Entropy Ratio Across the Shock')
    ax4.set_xlim(0, np.max(man_up_1**2) + 0.2)
    ax4.set_ylim(0.8, np.max(s_ratio_1) + 0.2)
    ax4.legend()
    ax4.grid()
    ax4.fill_between(x, ax1.get_ylim()[0], 1, color='gray', alpha=0.3)

    ## マッハ数比
    #ax5.plot(man_up**2, m_ratio, label=f'θ={theta_up}°, β={beta_up}', linewidth=0.8)
    #ax5.plot(point[1]**2, point2[0]**2/point2[1]**2, color='black', marker='o', markersize=3.8, label='Point 2')
    #ax5.set_xlabel('Upstream Mach Number')
    #ax5.set_ylabel('Mach Number Ratio M2 / M1')
    #ax5.set_title('Mach Number Ratio Across the Shock')
    #ax5.set_xlim(0, np.max(man_up_1**2) + 0.2)
    #ax5.set_ylim(0, 1.5*(point2[0]**2/point2[1]**2))
    #ax5.legend()
    #ax5.grid()

    plt.show()

def rh_analysis_graph_manual(
    beta_up,
    theta_up,
    point,
    n=2,
    xlim1=(0,3), ylim1=(0,3),
    xlim2=(0,3), ylim2=(-3,3),
    xlim3=(0,3), ylim3=(0,4),
    xlim4=(0,3), ylim4=(0.8,3)
):

    man_down = np.linspace(0.001, 20*point[0], 50000)
    man_up = np.vectorize(man_upstream)(theta_up, beta_up, man_down, gamma=5/3)

    tmp = man_up.copy()
    man_up = man_up[np.isfinite(tmp)]
    man_down = man_down[np.isfinite(tmp)]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 10))

    # ---------- Graph 1 ----------
    ax1.plot(man_down**2, man_up**2, label=f'θ={theta_up}°, β={beta_up}', linewidth=0.8)

    x = np.linspace(xlim1[0], xlim1[1], 200)
    ax1.plot(x, x, color='black', linewidth=0.3)

    ax1.plot(point[0]**2, point[1]**2, color='black', marker='o', markersize=3.8, label='Point')

    ax1.fill_between(x, 0, x, color='gray', alpha=0.3)

    ax1.set_xlabel('(Downstream Mach Number)^2')
    ax1.set_ylabel('(Upstream Mach Number)^2')

    ax1.set_xlim(*xlim1)
    ax1.set_ylim(*ylim1)

    ax1.axvline(x=1.0, color='black', linewidth=1.0)
    ax1.axhline(y=1.0, color='black', linewidth=1.0)

    ax1.set_title('Upstream vs Downstream Mach Number')
    ax1.legend()
    ax1.grid()

    # ---------- shock relations ----------
    Bt_ratio = (man_up**2 -1)/(man_down**2 -1)
    rho_ratio = (man_up**2)/(man_down**2)

    p_ratio = (
        1
        + 2 * man_up**2 * np.cos(np.radians(theta_up))**2 / beta_up * (1 - 1/rho_ratio)
        + np.tan(np.radians(theta_up))**2 * np.cos(np.radians(theta_up))**2 / beta_up * (1-Bt_ratio**2)
    )

    gamma = 5/3

    s_ratio = p_ratio / rho_ratio**gamma

    p_ratio_point = (
        1
        + 2 * point[1]**2 * np.cos(np.radians(theta_up))**2 / beta_up * (1 - 1/(point[1]**2/point[0]**2))
        + np.tan(np.radians(theta_up))**2 * np.cos(np.radians(theta_up))**2 / beta_up
        * (1-((point[1]**2 -1)/(point[0]**2 -1))**2)
    )

    s_ratio_point = p_ratio_point / (point[1]**2/point[0]**2)**gamma

    # ---------- Graph 2 ----------
    ax2.plot(man_up**2, -Bt_ratio, label=f'θ={theta_up}°, β={beta_up}', linewidth=0.8)

    ax2.plot(point[1]**2,
             -(point[1]**2 -1)/(point[0]**2 -1),
             color='black',
             marker='o',
             markersize=3.8)

    ax2.set_xlabel('(Upstream Mach Number)^2')
    ax2.set_ylabel('- Bt2 / Bt1')

    ax2.set_xlim(*xlim2)
    ax2.set_ylim(*ylim2)

    ax2.set_title('Ratio of Tangential Magnetic Fields')
    ax2.legend()
    ax2.grid()

    # ---------- Graph 3 ----------
    ax3.plot(man_up**2, rho_ratio, label=f'θ={theta_up}°, β={beta_up}', linewidth=0.8)

    ax3.plot(point[1]**2,
             (point[1]**2)/(point[0]**2),
             color='black',
             marker='o',
             markersize=3.8)

    ax3.set_xlabel('(Upstream Mach Number)^2')
    ax3.set_ylabel('Density Ratio ρ2 / ρ1')

    ax3.set_xlim(*xlim3)
    ax3.set_ylim(*ylim3)

    ax3.set_title('Density Ratio Across the Shock')
    ax3.legend()
    ax3.grid()

    ax3.fill_between(x, ylim3[0], 1, color='gray', alpha=0.3)

    # ---------- Graph 4 ----------
    ax4.plot(man_up**2, s_ratio, label=f'θ={theta_up}°, β={beta_up}', linewidth=0.8)

    ax4.plot(point[1]**2,
             s_ratio_point,
             color='black',
             marker='o',
             markersize=3.8)

    ax4.set_xlabel('(Upstream Mach Number)^2')
    ax4.set_ylabel('Entropy Ratio S2 / S1')

    ax4.set_xlim(*xlim4)
    ax4.set_ylim(*ylim4)

    ax4.set_title('Entropy Ratio Across the Shock')
    ax4.legend()
    ax4.grid()

    ax4.fill_between(x, ylim4[0], 1, color='gray', alpha=0.3)

    plt.show()


def rh_analysis_graph_arr(beta_up_arr, theta_up_arr, Ma_up_arr, Ma_down_arr, n=2):

    # 基準（プロット範囲用）
    theta_up = theta_up_arr[0]
    beta_up = beta_up_arr[0]
    point = [np.max(Ma_down_arr), np.max(Ma_up_arr)]

    man_down = np.linspace(0.001, 20*point[0], 50000)

    # 複数パラメータ向け upstream Mach array
    man_up = np.zeros((len(man_down), len(beta_up_arr)))
    for i in range(len(beta_up_arr)):
        man_up[:, i] = np.vectorize(man_upstream)(
            theta_up_arr[i], beta_up_arr[i], man_down, gamma=5/3
        )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 10))

    # -------------------------
    #  ax1: Upstream vs Downstream Mach Number
    # -------------------------
    for i in range(len(beta_up_arr)):
        line, = ax1.plot(man_down**2, man_up[:, i]**2, linewidth=0.8)
        ax1.plot(Ma_down_arr[i]**2, Ma_up_arr[i]**2,
                 marker='o', markersize=2.6, color=line.get_color())

    # 対角線
    x = np.linspace(0, np.max([np.max(Ma_down_arr)**2, np.max(Ma_up_arr)**2]) + 0.2, 200)
    ax1.plot(x, x, color='black', linewidth=0.3)

    # y < x の領域塗り
    ax1.fill_between(x, 0, x, color='gray', alpha=0.3)

    ax1.set_xlabel('(Downstream Mach Number)^2')
    ax1.set_ylabel('(Upstream Mach Number)^2')
    ax1.set_xlim(0, np.max(Ma_down_arr)**2 * 1.1)
    ax1.set_ylim(0, np.max(Ma_up_arr)**2 * 1.1)
    ax1.axvline(x=1.0, color='black', linestyle='-', linewidth=1.0)
    ax1.axhline(y=1.0, color='black', linestyle='-', linewidth=1.0)
    ax1.set_title('Upstream vs Downstream Mach Number')
    ax1.grid()

    # -------------------------
    #  ax2: −Bt2/Bt1
    # -------------------------
    for i in range(len(beta_up_arr)):

        Bt_ratio = (man_up[:, i]**2 - 1) / (man_down**2 - 1)
        line, = ax2.plot(man_up[:, i]**2, -Bt_ratio, linewidth=0.8)

        Bt_point = -(Ma_up_arr[i]**2 - 1) / (Ma_down_arr[i]**2 - 1)

        ax2.plot(Ma_up_arr[i]**2, Bt_point,
                 marker='o', markersize=2.6, color=line.get_color())

    ax2.set_xlabel('(Upstream Mach Number)^2')
    ax2.set_ylabel('- Bt2 / Bt1')
    ax2.set_xlim(0, np.max(Ma_up_arr)**2 * 1.1)
    # 全面負の場合、部分的に負の場合、全面性の場合
    if np.all(Bt_point < 0):
        ax2.set_ylim(np.min(Bt_point)*1.1, 0)
    elif np.any(Bt_point < 0):
        ax2.set_ylim(np.min(Bt_point)*1.1, np.max(Bt_point)*1.1)
    else:
        ax2.set_ylim(0, np.max(Bt_point)*1.1)
    ax2.set_title('Ratio of Tangential Magnetic Fields')
    ax2.grid()

    # -------------------------
    #  ax3: ρ2 / ρ1
    # -------------------------
    for i in range(len(beta_up_arr)):

        rho_ratio = man_up[:, i]**2 / man_down**2
        line, = ax3.plot(man_up[:, i]**2, rho_ratio, linewidth=0.8)

        rho_point = Ma_up_arr[i]**2 / Ma_down_arr[i]**2
        ax3.plot(Ma_up_arr[i]**2, rho_point,
                 marker='o', markersize=2.6, color=line.get_color())

    ax3.set_xlabel('(Upstream Mach Number)^2')
    ax3.set_ylabel('Density Ratio ρ2 / ρ1')
    ax3.set_xlim(0, np.max(Ma_up_arr)**2 * 1.1)
    ax3.set_ylim(0.8, np.max(rho_point)*1.02)
    ax3.set_title('Density Ratio Across the Shock')
    ax3.grid()

    # -------------------------
    #  ax4: S2 / S1
    # -------------------------
    gamma = 5/3
    for i in range(len(beta_up_arr)):

        rho_ratio = man_up[:, i]**2 / man_down**2
        Bt_ratio = (man_up[:, i]**2 - 1) / (man_down**2 - 1)

        p_ratio = (
            1
            + 2 * man_up[:, i]**2 * np.cos(np.radians(theta_up_arr[i]))**2 / beta_up_arr[i]
              * (1 - 1/rho_ratio)
            + np.tan(np.radians(theta_up_arr[i]))**2 * np.cos(np.radians(theta_up_arr[i]))**2 / beta_up_arr[i]
              * (1 - Bt_ratio**2)
        )

        s_ratio = p_ratio / rho_ratio**(gamma)

        line, = ax4.plot(man_up[:, i]**2, s_ratio, linewidth=0.8)

        # 点
        rho_pt = Ma_up_arr[i]**2 / Ma_down_arr[i]**2
        Bt_pt = (Ma_up_arr[i]**2 - 1) / (Ma_down_arr[i]**2 - 1)
        p_pt = (
            1
            + 2 * Ma_up_arr[i]**2 * np.cos(np.radians(theta_up_arr[i]))**2 / beta_up_arr[i]
              * (1 - 1/rho_pt)
            + np.tan(np.radians(theta_up_arr[i]))**2 * np.cos(np.radians(theta_up_arr[i]))**2 / beta_up_arr[i]
              * (1 - Bt_pt**2)
        )
        s_pt = p_pt / rho_pt**gamma

        ax4.plot(Ma_up_arr[i]**2, s_pt,
                 marker='o', markersize=2.6, color=line.get_color())

    ax4.set_xlabel('(Upstream Mach Number)^2')
    ax4.set_ylabel('Entropy Ratio S2 / S1')
    ax4.set_xlim(0, np.max(Ma_up_arr)**2 * 1.1)
    ax4.set_ylim(0.8, np.max(s_pt)*1.02)
    ax4.set_title('Entropy Ratio Across the Shock')
    ax4.grid()

    plt.tight_layout()
    plt.show()