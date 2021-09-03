import time,os,sys,getopt,subprocess as sb
import numpy as np
import time
from mpi4py import MPI
import h5py
import argparse
from numpy.random.mtrand import geometric

pi_by_180=np.pi/180
pi_by_360=np.pi/360
pi_by_180_inv=180/np.pi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs=comm.Get_size()
mask_nelems=361

def create_writer_list(nprocs,num_writers):
    return [i for i in range(0,nprocs,int(np.ceil(nprocs/num_writers)))]

def create_neighbours_list(rank,nprocs,writers):
    for i in range(len(writers[:-1])):
        if (rank >= writers[i]) and (rank < writers[i+1]):
            neighbours = [j for j in range(writers[i],writers[i+1])]
        elif rank >= writers[-1]:
            neighbours = [j for j in range(writers[-1],nprocs)]  
    return neighbours 
# Create communicator with writers

def writer_comm_create(writers):
# convinient to call collective for writing to database
    writer={'comm':MPI.COMM_NULL,
             'rank': None,
             'nprocs':None
             }
    group = comm.Get_group().Incl(writers)
    writer['comm'] = comm.Create(group)
    
    if rank in writers:
        writer['rank'] = writer['comm'].Get_rank()
        writer['nprocs'] = writer['comm'].Get_size()
        
    return writer

def create_domains(neighbours):
    
    local = dict()
    local = {'comm':MPI.COMM_NULL,
             'rank': None,
             'nprocs':None
             }
    
    group = comm.Get_group().Incl(neighbours)
    local['comm'] = comm.Create(group)
    if rank in neighbours:
        local['rank']  = local['comm'].Get_rank()
        local['nprocs']= local['comm'].Get_size()
    return local

# Create HDF5 and organize the required structure

def create_file(writer,fname,num_entries):
    
    width = mask_nelems
    ## still writing
    file_strcut = dict()
    fh = h5py.File(fname, 'w', driver='mpio', comm=writer['comm'])
    coordinates  = fh.create_group('pixels')
    geometry = fh.create_group('geometry')

    x_coords = coordinates.create_dataset('X', (num_entries,1), dtype='i')
    y_coords = coordinates.create_dataset('Y', (num_entries,1), dtype='i' )

    
    mask_h2  = geometry.create_dataset('mask_h2',(num_entries, width),dtype='f')
    mask_h2.attrs['height']=2
    mask_h10 = geometry.create_dataset('mask_h10',(num_entries, width),dtype='f')
    mask_h2.attrs['height']=10
    
    
    file_struct ={'handle': fh,
                  'x_coords': x_coords,
                  'y_coords': y_coords,
                  'mask_h2' : mask_h2,
                  'mask_h10': mask_h10
                   }
    return file_struct

def aggregate_posts(local,pixels,geometry):
    # Gather data on rank 0 of local communicator
    r_pixels    =   np.ndarray(shape=(local['nprocs'] * pixels.shape[0],pixels.shape[1]),
                                    dtype='i' )
    r_geometry  =   np.ndarray(shape=(local['nprocs'] * geometry.shape[0], 
                                      geometry.shape[1], geometry.shape[2]),
                                            dtype='f')
    
    local['comm'].Gather(pixels,r_pixels,root=0)
    local['comm'].Gather(geometry,r_geometry,root=0)
    return [r_pixels,r_geometry]

def write_data(y_iter,writer, fh, pixels,geometry):
    X  = fh['x_coords']
    Y  = fh['y_coords']
    m_h2 = fh['mask_h2']
    m_h10= fh['mask_h10']
    
    pixel_offset = y_iter  + writer['rank'] * pixels.shape[0]
    mask_offset =  y_iter  + writer['rank'] * geometry.shape[0]
    
    for i in range(pixels.shape[0]):
        X[i+pixel_offset] = pixels[i,0]
        Y[i+pixel_offset] = pixels[i,1]
        m_h2[i+mask_offset] = geometry[i,0]
        m_h10[i+mask_offset] = geometry[i,1]


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
    az= az* pi_by_180
    elev = elev * pi_by_180
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

def roundit(x):
     if (x-int(x)) >= 0.5:
         return(np.ceil(x))
     else:
         return(np.floor(x))

def fix_indx(x,y):
    if (x < y) & (y==0):
        return(1)
    elif (x > y) & (y!=0):
        return(y)
    else:
        return(x)

    
def getAltitude(lat,lon,num_samples,map_scale,scaling_factor,img):
    R = 2*1737400* np.tan((90+lat)*pi_by_360)
    x = R*np.sin(lon*pi_by_180)
    y = R*np.cos(lon*pi_by_180)
    #get Indices
    i=int(np.rint((x / map_scale + num_samples / 2 + 1 / 2))-1)
    i = fix_indx(i,0)
    i = fix_indx(i,num_samples - 1)

    j=int(np.rint((y / map_scale + num_samples / 2 + 1 / 2 ))+1)    
    j = fix_indx(j,0)
    j = fix_indx(j, num_samples - 1)
    #altitude
    alt = scaling_factor*img[i, j]
    return alt #, x, y, i, j


def getAltitude_deg(LAT_Q , LON_Q ,num_samples,map_scale,scaling_factor,img):
    return  [getAltitude(lat_q * pi_by_180_inv, lon_q * pi_by_180_inv,num_samples,map_scale,scaling_factor,img) for (lat_q,lon_q) in zip (LAT_Q,LON_Q)]

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
    long = np.arctan2(x,y) * pi_by_180_inv
    lat = -90 + pi_by_180_inv * 2*np.arctan(0.5 * R/1737400)
    return lat,long

def toLat(kx,ky,num_samples,map_scale):
    #to Cartesian
    x = (kx + 1 - num_samples / 2 - .5) * map_scale
    y = (ky - 1 - num_samples / 2 - .5) * map_scale
    R = np.sqrt(x*x + y*y)
    lat = -90 + pi_by_180_inv * 2*np.arctan(0.5 * R/1737400)
    return lat

def toLong(kx,ky,num_samples,map_scale):
    #to Cartesian
    x = (kx + 1 - num_samples / 2 - .5) * map_scale
    y = (ky - 1 - num_samples / 2 - .5) * map_scale
    R = np.sqrt(x*x + y*y)
    long = np.arctan2(x,y) * pi_by_180_inv
    return long


def calcMask(LAT_P, LON_P, h_P,num_samples,map_scale,scaling_factor,planetary_radius,reso,kmSurr,img,fName,kx,ky):
    Nh = len(h_P)
    #get Cartesian Coordinates
    ALT_P = getAltitude(LAT_P, LON_P,num_samples,map_scale,scaling_factor,img)
    P=np.array([0.0,0.0,0.0])
    P[0], P[1], P[2] = sph2cart(LON_P * pi_by_180, LAT_P * pi_by_180, planetary_radius + ALT_P)
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
        ALT_Q = [getAltitude(lat_q * pi_by_180_inv, lon_q * pi_by_180_inv,num_samples,map_scale,scaling_factor,img) for (lat_q,lon_q) in zip (LAT_Q,LON_Q)]
        #ALT_Q=list()
        #for lat_q,lon_q in zip (LAT_Q,LON_Q):
        #    ALT_Q.append(getAltitude(lat_q * pi_by_180_inv, lon_q * pi_by_180_inv,
        #                             num_samples,map_scale,scaling_factor,img) )
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
    parser = argparse.ArgumentParser(description='HLS script')
    parser.add_argument('--writers',required=True , type=int,
                        help='Number of writer MPI ranks to push data into database')
    parser.add_argument('--xrange',required=True, type=int,nargs=2,
                    help="number of x pixels [start,end]")
    parser.add_argument('--yrange',required=True,type=int,nargs=2,
                    help="number of y pixels [start,end]")
    args = parser.parse_args()
    
    num_writers = args.writers
    xrange = args.xrange
    yrange = args.yrange

    
    # Set up writers and aggreator domains and
    # instantiate databe and collection on writers
    writers = create_writer_list(nprocs,num_writers)
    neighbours = create_neighbours_list(rank,nprocs,writers)
    local = create_domains(neighbours)
    writer = writer_comm_create(writers)



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
    
    counting=0

    x_start =    xrange[0] + rank*my_xrange
    x_end   =    xrange[0] + (rank+1)*my_xrange
    if rank == (nprocs-1):
        x_end += rem
    mask    =   list()
    
    gather_time=0.0
    writer_time=0.0
    comm_time=0.0
    calc_time=0.0
    
    # Figure out how many rows will be written in HDF 5 file
    num_entries = np.empty(1,dtype='i')
    l_num_entries = (x_end - x_start) * (yrange[1] - yrange[0])
    num_entries=local['comm'].reduce(l_num_entries,op=MPI.SUM,root=0)
        
    if writer['comm'] != MPI.COMM_NULL:
        h5_fname='%s_x%s_%s_y%s_%s.h5' %(fName,xrange[0],xrange[1]-1,yrange[0],yrange[1]-1)
        fh=create_file(writer,h5_fname,writer['nprocs']*num_entries)
    l_pixels         = np.ndarray(shape=( (x_end-x_start), 2), dtype='i')
    l_geometry       = np.ndarray(shape=((x_end-x_start),2,mask_nelems), dtype='f')
    
    y_iter =0 
    for my in range(yrange[0],yrange[1]):
        s=time.time()
        ky=idy[my]
        calc_time=time.time()
        tmp_indx=0
        for mx in range( x_start,x_end):
            kx=idx[mx]
            
            l_pixels[tmp_indx,0] = kx 
            l_pixels[tmp_indx,1] = ky 
                # calculate whatever
            mask.append(calcMask(toLat(kx,ky,num_samples_g,map_scale_g),
                                 toLong(kx,ky,num_samples_g,map_scale_g),
                                 h_P,num_samples_g,map_scale_g,scaling_factor_g,planetary_radius_g,reso_g,kmSurr_g,img_g,fName,kx,ky))
            numpy_array= mask[-1] #last element of mask list
            
            l_geometry[tmp_indx,0] = numpy_array[0] 
            l_geometry[tmp_indx,1] = numpy_array[1] 
            tmp_indx +=1
            
        calc_time=time.time()-calc_time
        comm.Barrier()
        
        comm_time=time.time()
        
        gather_time = time.time()
        if local['comm'] != MPI.COMM_NULL:
            posts=aggregate_posts(local,l_pixels,l_geometry)
        gather_time = time.time() -gather_time
        
        writer_time=time.time()
        if writer['comm'] != MPI.COMM_NULL:
            write_data(y_iter*total_pixl_x,writer, fh, posts[0],posts[1])
        writer_time=time.time()-writer_time
       
        comm_time=time.time()-comm_time
        y_iter +=1
        print("rank:",rank," rem:", rem," xrange:",xrange," yrange:", yrange, " total_pixels:",total_pixl_x," my_xrange:",my_xrange,
              " start,end:",x_start,x_end, " kx,ky:",kx,ky,"mx,my:",mx,my,
              "elapsed:",(time.time()-s),'calc:',calc_time,'comm:',comm_time,'gather:',gather_time,
              'writer_time:',writer_time)

        
    if writer['comm'] != MPI.COMM_NULL:
        fh['handle'].close()

                

if __name__ == "__main__":
    main(sys.argv[1:])
