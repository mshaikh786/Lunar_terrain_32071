import time,os,sys,getopt,subprocess as sb
import numpy as np
import time
from pymongo import MongoClient

#---------------------
host=sb.run(['cat','db_host'],stdout=sb.PIPE)
host=host.stdout.decode('utf-8').strip()
db_name="test"
collection_name="LDEM_80S_20M"
client = MongoClient(host)
#---------------------

#outdir='data_mpi'
#os.makedirs(outdir,exist_ok=True)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs=comm.Get_size()

def sph2cart(az,elev,r):
    z = r * np.sin(elev)
    rcoselev = r * np.cos(elev)
    x = rcoselev * np.cos(az)
    y = rcoselev * np.sin(az)
    return x,y,z


def sph2cart_all(LAT_Q,LON_Q,ALT_Q,r_planetary=1737400):
    return [sph2cart(lon_q,lat_q,r_planetary + alt_q) for (lat_q,lon_q,alt_q) in zip (LAT_Q,LON_Q,ALT_Q)]

def planetary_sph2cart(az, elev ,alt,r_planetary=1737400):
    r=r_planetary+alt
    az= az* np.pi / 180
    elev = elev * np.pi / 180
    rcoselev = r * np.cos(elev)
    x = rcoselev * np.cos(az)
    y = rcoselev * np.sin(az)
    z = r * np.sin(elev)
    return x,y,z

def cart2sph(x,y,z):
    hypotxy = np.hypot(x, y)
    r = np.hypot(hypotxy, z)
    elev = np.arctan2(z, hypotxy)
    az = np.arctan2(y, x)
    return az,elev,r

def cart2sph_long(x,y,z):
    az = np.arctan2(y, x)
    return az

def cart2sph_lat(x,y,z):
    hypotxy = np.hypot(x, y)
    elev = np.arctan2(z, hypotxy)
    return elev

def getAltitude(lat,lon,num_samples,map_scale,scaling_factor,img):
    R = 2*1737400* np.tan((90+lat)*np.pi/360)
    x = R*np.sin(lon*np.pi/180)
    y = R*np.cos(lon*np.pi/180)
    #get Indices
    i=int(np.round((x / map_scale + num_samples / 2 + 1 / 2))-1)
    i = np.where(i < 0, 1, i)
    i = np.where(i > num_samples - 1, num_samples - 1, i)
    j=int(np.round((y / map_scale + num_samples / 2 + 1 / 2 ))+1)
    j = np.where(j < 0, 1, j)
    j = np.where(j > num_samples - 1, num_samples - 1, j)
    #altitude
    alt = scaling_factor*img[i, j]
    return alt #, x, y, i, j


def getAltitude_deg(LAT_Q , LON_Q ,num_samples,map_scale,scaling_factor,img):
    return  [getAltitude(lat_q * 180 / np.pi, lon_q * 180 / np.pi,num_samples,map_scale,scaling_factor,img) for (lat_q,lon_q) in zip (LAT_Q,LON_Q)]

def getIdx(X,Y,num_samples,map_scale):
    i=np.array([  int((x/map_scale + num_samples/2 + 1/2))-1 for x in X])
    # i(i<1)=1
    i[ i < 1 ] = 1
    i[ i > num_samples] = num_samples
    j=np.array([ int((y / map_scale + num_samples / 2 + 1 / 2 ))+1 for y in Y])
    j[ j < 1] = 1
    j[ j > num_samples] = num_samples

    return i.astype(int),j.astype(int)

def toLatLong(kx,ky,num_samples,map_scale):
    #to Cartesian
    x = (kx + 1 - num_samples / 2 - .5) * map_scale
    y = (ky - 1 - num_samples / 2 - .5) * map_scale
    R = np.sqrt(x*x + y*y)
    long = np.arctan2(x,y) * 180/np.pi
    lat = -90 + 180/np.pi * 2*np.arctan(0.5 * R/1737400)
    return lat,long

def toLat(kx,ky,num_samples,map_scale):
    #to Cartesian
    x = (kx + 1 - num_samples / 2 - .5) * map_scale
    y = (ky - 1 - num_samples / 2 - .5) * map_scale
    R = np.sqrt(x*x + y*y)
    lat = -90 + 180/np.pi * 2*np.arctan(0.5 * R/1737400)
    return lat

def toLong(kx,ky,num_samples,map_scale):
    #to Cartesian
    x = (kx + 1 - num_samples / 2 - .5) * map_scale
    y = (ky - 1 - num_samples / 2 - .5) * map_scale
    R = np.sqrt(x*x + y*y)
    long = np.arctan2(x,y) * 180/np.pi
    return long


def calcMask(LAT_P, LON_P, h_P,num_samples,map_scale,scaling_factor,planetary_radius,reso,kmSurr,img,fName,kx,ky):
    Nh = len(h_P)
    #get Cartesian Coordinates
    ALT_P = getAltitude(LAT_P, LON_P,num_samples,map_scale,scaling_factor,img)
    P=np.array([0.0,0.0,0.0])
    P[0], P[1], P[2] = sph2cart(LON_P * np.pi / 180, LAT_P * np.pi / 180, planetary_radius + ALT_P)
    P_hat=P/np.linalg.norm(P)
    N=np.array([0.0,0.0,1.0])
    P_E=np.cross(N,P_hat)
    P_E = P_E / np.linalg.norm(P_E)
    P_N = np.cross(P_hat, P_E)
    P_N = P_N / np.linalg.norm(P_N)
    d =  np.arange(reso,int((1000*kmSurr))+1,reso)#xo: dx: xf
    ud = np.ones(len(d))
    Az = np.arange(0,361)
    nAz = len(Az);
    mask = np.zeros((Nh, nAz))
    howFar = np.zeros((Nh, nAz))
    for AzN in range(0,nAz):
        Q=np.outer(ud,P) + np.outer(d,np.sin(np.deg2rad(Az[AzN])) * P_N) + np.outer(d, np.cos(np.deg2rad(Az[AzN])) * P_E) #Q = ud*P + d * sind(Az(AzN)) * P_N + d * cosd(Az(AzN)) * P_E;%Az=0 at East and 90 at North
        #QQ= [cart2sph(q[0], q[1], q[2]) for q in Q]
        LON_Q = [cart2sph_long(q[0], q[1], q[2]) for q in Q] #np.asarray(list(zip(*QQ))[0])
        LAT_Q = [cart2sph_lat(q[0], q[1], q[2]) for q in Q]#np.asarray(list(zip(*QQ))[1])
        ALT_Q = [getAltitude(lat_q * 180 / np.pi, lon_q * 180 / np.pi,num_samples,map_scale,scaling_factor,img) for (lat_q,lon_q) in zip (LAT_Q,LON_Q)]
        Q=[sph2cart(lon_q,lat_q,planetary_radius + alt_q) for (lat_q,lon_q,alt_q) in zip (LAT_Q,LON_Q,ALT_Q)]
        Q=np.asarray(Q)
        for h0N in range(0,Nh):
            P = P_hat * (planetary_radius + ALT_P + h_P[h0N])
            PQ = Q-np.outer(ud , P)
            PQ = PQ / np.outer(np.sqrt(np.sum(PQ*PQ, 1)) , np.ones((1, 3)) )
            Beta = np.rad2deg(np.arccos(np.sum(PQ * ( np.outer(ud , P_hat)) , 1) ) )
            howFar[h0N,AzN]=np.argmax(90-Beta)
            mask[h0N, AzN] = 90 - Beta[int(howFar[h0N, AzN]) ]
#    np.savetxt(outdir + '/' + fName + '_' + str(kx) + '_' + str(ky) + '_%d_mask.csv' %rank  , mask, delimiter=',',fmt='%f')
#    np.savetxt(outdir + '/' + fName + '_' + str(kx) + '_' + str(ky) + '_%d_howFar.csv' %rank  , howFar, delimiter=',',fmt='%i')
    return mask

def norm(P):
    return P/np.linalg.norm(P)

def cross(a,b):
    return np.cross(a,b)

def d_planetocentric(Az_idx,d,ud,P,P_N,P_E):
    b=np.outer(d,np.sin(np.deg2rad(Az[AzN])) * P_N)
    a=np.outer(d, np.cos(np.deg2rad(Az[AzN])) * P_E)
    c=np.outer(ud,P)
    return a + b + c

def eval_LON(Q):
    return [cart2sph_long(q[0], q[1], q[2]) for q in Q]

def eval_LAT(Q):
    return [cart2sph_lat(q[0], q[1], q[2]) for q in Q]

def eval_P2(P_hat,planetary_radius,ALT_P,h_P_idx):
    return  P_hat * (planetary_radius + ALT_P + h_P_idx)

def eval_PQ_norm(Q2,ud,P2):
    PQ = Q2-np.outer(ud , P2)
    PQ = PQ / np.outer(np.sqrt(np.sum(PQ*PQ, 1)) , np.ones((1, 3)) )
    return PQ

def eval_Beta(PQ,ud,P_hat):
    return np.rad2deg(np.arccos(np.sum(PQ * ( np.outer(ud , P_hat)) , 1) ) )

def eval_HowFar(Beta):
    return np.argmax(90-Beta)

def eval_mask(Beta,howFar,h0N,AzN):
    return 90 - Beta[int(howFar[h0N, AzN]) ]

def main(argv):


    xrange = []
    yrange = []
    try:
        opts, args = getopt.getopt(argv, "hx:y:H:", ["xrange=", "yrange=", "hostname="])
    except getopt.GetoptError:
        print('HLS_BuildMask.py -x [start,end] -y [start,end] -H <host ip or name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('HLS_BuildMask.py -x [start,end] -y [start,end] -H <host ip or name')
            sys.exit()
        elif opt in ("-x", "--xrange"):
            xrange = eval(arg)
        elif opt in ("-y", "--yrange"):
            yrange = eval(arg)
        elif opt in ("-H", "--hostname"):
            hostname = arg
    #print('x range is ', xrange)
    #print('y range is ', yrange)


    #globals
    reso_g = 20
    kmSurr_g = 150
    num_samples_g = 30400  # TODO:  from LBL
    map_scale_g = 20  # TODO: from LBL
    scaling_factor_g = 0.5000 # TODO: from LBL
    planetary_radius_g = 1737400  # TODO: from LBL
    img_g=[]
    nx = 30400  # TODO: get this from LBL
    # Parameters.
    input_filename = "80S_20mpp/LDEM_80S_20M.IMG"
    shape = (nx, nx)  # matrix size
    dtype = np.dtype('int16')  # little-endian signed integer (16bit)

    output_filename = "LDEM_80S_20M.PNG"
    
    # Reading.
    fid = open(input_filename, 'rb')
    data = np.fromfile(fid, dtype)
    data = data.reshape(shape, order='F')  # use order='F' so row column order corresponds to Matlab

    data = data.reshape(shape)
    data = data.T
    img_g = np.rot90(data, -1)
    
    #---------------------
    if rank == 0:
        db    = client[db_name]
        col   = db[collection_name]
    comm.Barrier()
    if rank > 0:
        db = client[db_name]
        col = db[collection_name] 
    print("[%s]:"% rank,db)
    #----------------------
    
    h_P = np.array([2,10])

    # construct a grid of interest, points are wrt lunar south pole
    xo = -304e3
    xf = 304e3
    yo = -304e3
    yf = 304e3
    dx = 1 * reso_g
    dy = 1 * reso_g
    locName = 'SpudisRidge'
    X = range(int(xo),int(xf)+1,int(dx))#xo: dx: xf
    Y = range(int(yo),int(yf)+1,int(dy)) #yo: dy: yf
    idx, idy = getIdx(X, Y,num_samples_g,map_scale_g)

    fName = 'LDEM_80S_20M'

    calcmask_futures=[]
    calcmask_futures_list=[]
    
    total_pixl_x=xrange[1]- xrange[0]
    
    rem=total_pixl_x % nprocs
    my_xrange=total_pixl_x//nprocs
    
    

    x_start =    xrange[0] + rank*my_xrange
    x_end   =    xrange[0] + (rank+1)*my_xrange
    if rank == (nprocs-1):
        x_end += rem
    mask    =   list()
    for my in range(yrange[0],yrange[1]):
        s=time.time()
        ky=idy[my]
        for mx in range( x_start,x_end):
            kx=idx[mx]
            post = {}
            post['coordinates'] = [int(kx), int(ky)]
            result1 = db[collection_name].update_one({"coordinates": post['coordinates']}, {"$set": post}, upsert=True)
            if result1.matched_count == 0:  # this is a new entry
                # calculate whatever
                mask.append(calcMask(toLat(kx,ky,num_samples_g,map_scale_g),
                                     toLong(kx,ky,num_samples_g,map_scale_g),
                                     h_P,num_samples_g,map_scale_g,scaling_factor_g,planetary_radius_g,reso_g,kmSurr_g,img_g,fName,kx,ky))
                numpy_array= mask[-1] #last element of mask list
                post['geometry'] = [{'height': 2, 'mask': numpy_array[0].tolist()}, {'height': 10, 'mask': numpy_array[1].tolist()}]
                db[collection_name].update_one({"coordinates": post['coordinates']}, {"$set": post}, upsert=True)
        print("rank:",rank," rem:", rem," xrange:",xrange," yrange:", yrange, " total_pixels:",total_pixl_x," my_xrange:",my_xrange," start,end:",x_start,x_end, " kx,ky:",kx,ky,"mx,my:",mx,my,"elapsed:",(time.time()-s))

if __name__ == "__main__":
    main(sys.argv[1:])
