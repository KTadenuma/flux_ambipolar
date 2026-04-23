import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import openmhd
import gc
import cv2

def plot_2d_reconnection(file_name, mode, variable):
    # dummy index
    vx=0;vy=1;vz=2;pr=3;ro=4;bx=5;by=6;bz=7;ps=8
    # reading the data ...
    # x,y,t,data = openmhd.data_read("data/field-00010.dat")
    # reading the data (subdomain: [ix1,ix2] x [jx1,jx2] or xrange (x1,x2) x yrange (y1,y2))
    # x,y,t,data = openmhd.data_read("data/field-00010.dat",ix1=0,ix2=1301,jx1=0,jx2=151)
    # x,y,t,data = openmhd.data_read("data/field-00010.dat",ix1=0,ix2=3901,jx1=0,jx2=451) # Zenitani & Miyoshi 2011 [6000 x 4500]
    x,y,t,data = openmhd.data_read(file_name,xrange=(0.0,130.0),yrange=(0.0,15.0))

    # 2D mirroring (This depends on the BC)
    ix = x.size
    jx = 2*y.size-2
    jxh= y.size-1
    tmp  = data
    data = np.ndarray((ix,jx,9),np.double)
    data[:,jxh:,:]   =  tmp[:,1:,:]
    data[:,0:jxh, :] =  tmp[:,-1:-jxh-1:-1, :]
    data[:,0:jxh,vy] = -tmp[:,-1:-jxh-1:-1,vy]
    data[:,0:jxh,vz] = -tmp[:,-1:-jxh-1:-1,vz]
    data[:,0:jxh,bx] = -tmp[:,-1:-jxh-1:-1,bx]
    data[:,0:jxh,ps] = -tmp[:,-1:-jxh-1:-1,ps]
    # releasing the memory, because this tmp could be large
    del tmp
    gc.collect()

    tmp = y
    y = np.ndarray((jx),np.double)
    y[jxh:]  =  tmp[1:]
    y[0:jxh] = -tmp[-1:-jxh-1:-1]

    # preparing the canvas
    fig = plt.figure(figsize=(10, 5), dpi=80)
    # fig.clear()
    plt.clf()

    # extent: [left, right, bottom, top]
    extent=[x[0],x[-1],y[0],y[-1]]
    # 2D plot (vmin/mymin: minimum value, vmax/mymax: max value)
    # Note: ().T is necessary for 2-D plot routines (imshow/pcolormesh...)
    tmp = np.ndarray((x.size,y.size),np.double)
    tmp[:,:] = data[:,:,variable]
    data_full = data
    if mode == 0:
        cmap ="jet"
    elif mode == 1:
        tmp = cv2.Laplacian(tmp, -1, ksize=3)
        binary = np.zeros_like(tmp)
        _, binary = cv2.threshold(np.abs(tmp), 0.05*tmp.max(), 255, cv2.THRESH_BINARY) 
        tmp = binary
        cmap ="gray"
    elif mode == 2:
        # tmp が float の場合 → 正規化して uint8 に変換
        tmp_uint8 = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        tmp = cv2.Canny(tmp_uint8, 10.0, 20.0)
        cmap ="gray"
    elif mode == 3:
        tmp = cv2.Sobel(tmp, -1, 1, 0, ksize=3)
        binary = np.zeros_like(tmp)
        _, binary = cv2.threshold(np.abs(tmp), 0.07*tmp.max(), 255, cv2.THRESH_BINARY)
        tmp = binary
        cmap ="gray"
    mymax = max(tmp.max(), -tmp.min()) if( tmp.max() > 0.0 ) else 0.0
    mymin = min(tmp.min(), -tmp.max()) if( tmp.min() < 0.0 ) else 0.0
    if mode == 2:
        mymin = -100
        mymax = 255
    myimg = plt.imshow(tmp.T,origin='lower',vmin=mymin,vmax=mymax,cmap=cmap,extent=extent,aspect='auto')

    # image operations (e.g. colormaps)
    # myimg.set_cmap('jet')
    # myimg.set_cmap('RdBu_r')  # colortable(70,/reverse) in IDL
    # myimg.set_cmap('seismic')
    # myimg.set_cmap('bwr')
    # myimg.set_cmap('gist_ncar_r')
    # myimg.set_cmap('Pastel1')

    # useful options
    # plt.grid()
    plt.xlabel("X",size=16)
    plt.ylabel("Y",size=16)
    plt.title('Outflow speed (t = %6.1f)' % t, size=20)
    # colorbar
    plt.colorbar()

    # preparing Vector potential (Az)
    if mode == 0:
        az = np.ndarray((x.size,y.size),np.double)
        fx = 0.5*(x[1]-x[0])
        fy = 0.5*(y[1]-y[0])
        az[0,0] = (fy*data[0,0,bx] - fx*data[0,0,by])
        for j in range(1,y.size):
            az[0,j] = az[0,j-1] + fy*(data[0,j-1,bx]+data[0,j,bx])
        for i in range(1,x.size):
            az[i,:] = az[i-1,:] - fx*(data[i-1,:,by]+data[i,:,by])

        # contour of Az = magnetic field lines
        plt.contour(az.T,extent=extent,colors='w',linestyles='solid')

    # plot
    plt.show()

    # adjusting the margins...
    plt.tight_layout()

    # image file
    # plt.savefig('output.png', dpi=80)
    return x, y, t, tmp[:, :], data_full

def plot_2d_reconnection_simplified(file_name, mode, variable, figure=False):
    # dummy index
    vx=0;vy=1;vz=2;pr=3;ro=4;bx=5;by=6;bz=7;ps=8
    # reading the data ...
    # x,y,t,data = openmhd.data_read("data/field-00010.dat")
    # reading the data (subdomain: [ix1,ix2] x [jx1,jx2] or xrange (x1,x2) x yrange (y1,y2))
    # x,y,t,data = openmhd.data_read("data/field-00010.dat",ix1=0,ix2=1301,jx1=0,jx2=151)
    # x,y,t,data = openmhd.data_read("data/field-00010.dat",ix1=0,ix2=3901,jx1=0,jx2=451) # Zenitani & Miyoshi 2011 [6000 x 4500]
    x,y,t,data = openmhd.data_read(file_name,xrange=(0.0,130.0),yrange=(0.0,15.0))

    # 2D mirroring (This depends on the BC)
    ix = x.size
    jx = 2*y.size-2
    jxh= y.size-1
    tmp  = data
    data = np.ndarray((ix,jx,9),np.double)
    data[:,jxh:,:]   =  tmp[:,1:,:]
    data[:,0:jxh, :] =  tmp[:,-1:-jxh-1:-1, :]
    data[:,0:jxh,vy] = -tmp[:,-1:-jxh-1:-1,vy]
    data[:,0:jxh,vz] = -tmp[:,-1:-jxh-1:-1,vz]
    data[:,0:jxh,bx] = -tmp[:,-1:-jxh-1:-1,bx]
    data[:,0:jxh,ps] = -tmp[:,-1:-jxh-1:-1,ps]
    # releasing the memory, because this tmp could be large
    del tmp
    gc.collect()

    tmp = y
    y = np.ndarray((jx),np.double)
    y[jxh:]  =  tmp[1:]
    y[0:jxh] = -tmp[-1:-jxh-1:-1]

    if figure == True:

        # preparing the canvas
        fig = plt.figure(figsize=(10, 5), dpi=80)
        # fig.clear()
        plt.clf()

        # extent: [left, right, bottom, top]
        extent=[x[0],x[-1],y[0],y[-1]]
    # 2D plot (vmin/mymin: minimum value, vmax/mymax: max value)
    # Note: ().T is necessary for 2-D plot routines (imshow/pcolormesh...)
    tmp = np.ndarray((x.size,y.size),np.double)
    tmp[:,:] = data[:,:,variable]
    data_full = data
    if mode == 0:
        cmap ="jet"
    elif mode == 1:
        tmp = cv2.Laplacian(tmp, -1, ksize=3)
        binary = np.zeros_like(tmp)
        _, binary = cv2.threshold(np.abs(tmp), 0.05*tmp.max(), 255, cv2.THRESH_BINARY) 
        tmp = binary
        cmap ="gray"
    elif mode == 2:
        # tmp が float の場合 → 正規化して uint8 に変換
        tmp_uint8 = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        tmp = cv2.Canny(tmp_uint8, 10.0, 20.0)
        cmap ="gray"
    elif mode == 3:
        tmp = cv2.Sobel(tmp, -1, 1, 0, ksize=3)
        binary = np.zeros_like(tmp)
        _, binary = cv2.threshold(np.abs(tmp), 0.07*tmp.max(), 255, cv2.THRESH_BINARY)
        tmp = binary
        cmap ="gray"
    mymax = max(tmp.max(), -tmp.min()) if( tmp.max() > 0.0 ) else 0.0
    mymin = min(tmp.min(), -tmp.max()) if( tmp.min() < 0.0 ) else 0.0
    if figure == True:
        if mode == 2:
            mymin = -100
            mymax = 255
            myimg = plt.imshow(tmp.T,origin='lower',vmin=mymin,vmax=mymax,cmap=cmap,extent=extent,aspect='auto')

        plt.xlabel("X",size=16)
        plt.ylabel("Y",size=16)
        plt.title('Outflow speed (t = %6.1f)' % t, size=20)
        # colorbar
        plt.colorbar()

    # preparing Vector potential (Az)
        if mode == 0:
            az = np.ndarray((x.size,y.size),np.double)
            fx = 0.5*(x[1]-x[0])
            fy = 0.5*(y[1]-y[0])
            az[0,0] = (fy*data[0,0,bx] - fx*data[0,0,by])
            for j in range(1,y.size):
                az[0,j] = az[0,j-1] + fy*(data[0,j-1,bx]+data[0,j,bx])
            for i in range(1,x.size):
                az[i,:] = az[i-1,:] - fx*(data[i-1,:,by]+data[i,:,by])

            # contour of Az = magnetic field lines
            plt.contour(az.T,extent=extent,colors='w',linestyles='solid')

        # plot
        plt.show()

        # adjusting the margins...
        plt.tight_layout()

    # image file
    # plt.savefig('output.png', dpi=80)
    return x, y, t, tmp[:, :], data_full

def plot_2d_reconnection_variable(x, y, t, data, mode):
    # dummy index
    vx=0;vy=1;vz=2;pr=3;ro=4;bx=5;by=6;bz=7;ps=8

    # preparing the canvas
    fig = plt.figure(figsize=(10, 5), dpi=80)
    # fig.clear()
    plt.clf()

    # extent: [left, right, bottom, top]
    extent=[x[0],x[-1],y[0],y[-1]]
    # 2D plot (vmin/mymin: minimum value, vmax/mymax: max value)
    # Note: ().T is necessary for 2-D plot routines (imshow/pcolormesh...)
    tmp = np.ndarray((x.size,y.size),np.double)
    tmp[:,:] = data[:,:]
    data_full = data
    if mode == 0:
        cmap ="jet"
    elif mode == 1:
        tmp = cv2.Laplacian(tmp, -1, ksize=3)
        binary = np.zeros_like(tmp)
        _, binary = cv2.threshold(np.abs(tmp), 0.05*tmp.max(), 255, cv2.THRESH_BINARY) 
        tmp = binary
        cmap ="gray"
    elif mode == 2:
        # tmp が float の場合 → 正規化して uint8 に変換
        tmp_uint8 = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        tmp = cv2.Canny(tmp_uint8, 8.0, 18.0)
        cmap ="gray"
    elif mode == 3:
        tmp = cv2.Sobel(tmp, -1, 1, 0, ksize=3)
        binary = np.zeros_like(tmp)
        _, binary = cv2.threshold(np.abs(tmp), 0.07*tmp.max(), 255, cv2.THRESH_BINARY)
        tmp = binary
        cmap ="gray"
    mymax = max(tmp.max(), -tmp.min()) if( tmp.max() > 0.0 ) else 0.0
    mymin = min(tmp.min(), -tmp.max()) if( tmp.min() < 0.0 ) else 0.0
    if mode == 2:
        mymin = -100
        mymax = 255
    myimg = plt.imshow(tmp.T,origin='lower',vmin=mymin,vmax=mymax,cmap=cmap,extent=extent,aspect='auto')

    # useful options
    # plt.grid()
    plt.xlabel("X",size=16)
    plt.ylabel("Y",size=16)
    plt.title('Outflow speed (t = %6.1f)' % t, size=20)
    # colorbar
    plt.colorbar()
    plt.show()

    # adjusting the margins...
    plt.tight_layout()

    # image file
    # plt.savefig('output.png', dpi=80)
    return x, y, t, tmp[:, :], data_full

def plot_section(x, y, t, i1, i2, j1, j2, data, cmap):
    print("Location", x[i1], x[i2], y[j1], y[j2])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x_tmp = x[i1:i2]
    y_tmp = y[j1:j2]
    data_tmp = data[i1:i2, j1:j2]
    myimg_1 = ax1.imshow(
        data_tmp.T,
        origin='lower',
        extent=[x[i1], x[i2], y[j1], y[j2]],
        aspect='auto',
        cmap=cmap
    )
    ax1.set_xlabel("X", size=16)
    ax1.set_ylabel("Y", size=16)
    ax1.set_title('Section', size=20)
    fig.colorbar(myimg_1, ax=ax1)

    # --- カラースケールを ±対称にする ---
    data_float = np.array(data, dtype=float)  # 安全のためfloat変換
    mymax = np.nanmax(np.abs(data_float))
    myimg_2 = ax2.imshow(
        data_float.T,
        origin='lower',
        cmap=cmap,
        vmin=-mymax,
        vmax=mymax,
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect='auto'
    )
    fig.colorbar(myimg_2, ax=ax2)

    ax2.set_xlabel("X", size=16)
    ax2.set_ylabel("Y", size=16)
    ax2.set_title(f't = {t:6.1f}', size=20)

    rect = patches.Rectangle(
        (x[i1], y[j1]),
        x[i2] - x[i1],
        y[j2] - y[j1],
        linewidth=2,
        edgecolor='k',
        facecolor='none'
    )
    ax2.add_patch(rect)

    plt.tight_layout()
    plt.show()
    return x_tmp, y_tmp, data_tmp

def plot_section_simplified(x, y, t, i1, i2, j1, j2, data, cmap, figure):
    # print("Location", x[i1], x[i2], y[j1], y[j2])
    x_tmp = x[i1:i2]
    y_tmp = y[j1:j2]
    data_tmp = data[i1:i2, j1:j2]

    if (figure == True):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        myimg_1 = ax1.imshow(
            data_tmp.T,
            origin='lower',
            extent=[x[i1], x[i2], y[j1], y[j2]],
            aspect='auto',
            cmap=cmap
        )
        ax1.set_xlabel("X", size=16)
        ax1.set_ylabel("Y", size=16)
        ax1.set_title('Section', size=20)
        fig.colorbar(myimg_1, ax=ax1)

        # --- カラースケールを ±対称にする ---
        data_float = np.array(data, dtype=float)  # 安全のためfloat変換
        mymax = np.nanmax(np.abs(data_float))
        myimg_2 = ax2.imshow(
            data_float.T,
            origin='lower',
            cmap=cmap,
            vmin=-mymax,
            vmax=mymax,
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect='auto'
        )
        fig.colorbar(myimg_2, ax=ax2)

        ax2.set_xlabel("X", size=16)
        ax2.set_ylabel("Y", size=16)
        ax2.set_title(f't = {t:6.1f}', size=20)

        rect = patches.Rectangle(
            (x[i1], y[j1]),
            x[i2] - x[i1],
            y[j2] - y[j1],
            linewidth=2,
            edgecolor='k',
            facecolor='none'
        )
        ax2.add_patch(rect)

        plt.tight_layout()
        plt.show()
    return x_tmp, y_tmp, data_tmp

def data_info(x, y, data_full, i1, i2, j1, j2, variable, mode):
    x_needed = x[i1:i2]
    y_needed = y[j1:j2]
    data = data_full
    if mode == 0:
        data_needed = data[i1:i2, j1:j2, variable]
    elif mode == 1:
        data_tmp = cv2.Laplacian(data[:, :, variable], -1, ksize=3)
        data_needed = data_tmp[i1:i2, j1:j2]
    elif mode == 2:
        # tmp が float の場合 → 正規化して uint8 に変換
        tmp_uint8 = cv2.normalize(data[:, :, variable], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        data_tmp = cv2.Canny(tmp_uint8, 10.0, 20.0)
        data_needed = data_tmp[i1:i2, j1:j2]

    check = True
    if np.sum(data_needed) == 0.0:
        check = False
        
    return x_needed, y_needed, data_needed, check

def i_optimize(data, i_start, i_end, j, small_box=False, step=3):
    box = []
    for i in range(i_start, i_end+1, step):
        j1=j-10 ; j2=j+10 ; i1=i-10 ; i2=i+10 # Petschek shock
        if small_box:
            j1=j-5 ; j2=j+5 ; i1=i-5 ; i2=i+5
        tmp = data[i1:i2, j1:j2]
        if np.any(tmp > 0):
            box.append(i)

    box = np.array(box)
    i_optimal = np.mean(box).astype(int)
    j_optimal = j
    return i_optimal, j_optimal

def j_optimize(data, i, j_start, j_end, small_box=False, step=3):
    box = []
    for j in range(j_start, j_end+1, step):
        j1=j-10 ; j2=j+10 ; i1=i-10 ; i2=i+10 # Petschek shock
        if small_box:
            j1=j-5 ; j2=j+5 ; i1=i-5 ; i2=i+5
        tmp = data[i1:i2, j1:j2]
        if np.any(tmp > 0):
            box.append(j)

    box = np.array(box)
    j_optimal = np.mean(box).astype(int)
    i_optimal = i
    return i_optimal, j_optimal
