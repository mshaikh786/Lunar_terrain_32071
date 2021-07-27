import time,os,sys,getopt,subprocess as sb
from pymongo import MongoClient
from mpi4py import MPI
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Stats')
parser.add_argument('--writers',required=True , type=int,
                    help='Number of writer MPI ranks to push data into database')
args = parser.parse_args()
num_writers = args.writers

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()



# Create communicator with writers
def writer_comm_create(writers):
# convinient to call collective for writing to database
    writer={'comm':MPI.COMM_NULL,
             'rank': None,
             'nprocs':None}
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
             'nprocs':None}
    
    group = comm.Get_group().Incl(neighbours)
    local['comm'] = comm.Create(group)
    if rank in neighbours:
        local['rank']  = local['comm'].Get_rank()
        local['nprocs']= local['comm'].Get_size()
    return local

# Instantiate database on write ranks only

def db_connect(writer):
    
        host=sb.run(['cat','db_host'],stdout=sb.PIPE)
        host=host.stdout.decode('utf-8').strip()
        db_name="test"
        collection_name="LDEM_80S_20M"
        client = MongoClient(host)
        
        
        db_info = {'host':host,
                   'db_name':db_name,
                   'collection_name':collection_name,
                   'db':None,
                   'collection':None
            }
    
        if writer['rank'] is 0:
            db_info['db']   = client[db_info['db_name']]
            db_info['coll'] = db_info['db'][db_info['collection_name']]
        writer['comm'].Barrier()
        db_info['db'] = client[db_info['db_name']]
        db_info['coll']= db_info['db'][db_info['collection_name']]
        return db_info



#def post(writer,db):
    


if __name__ == "__main__":
  
    writers = [i for i in range(0,nprocs,int(np.ceil(nprocs/num_writers)))]
    
    for i in range(len(writers[:-1])):
        if (rank >= writers[i]) and (rank < writers[i+1]):
            neighbours= [j for j in range(writers[i],writers[i+1])]
        elif rank >= writers[-1]:
            neighbours = [j for j in range(writers[-1],nprocs)]    
        
    local = create_domains(neighbours)
    print(rank,local['rank']," has",neighbours[0],neighbours[-1])
    writer = writer_comm_create(writers)
    if writer['comm'] != MPI.COMM_NULL:
        db_info = db_connect(writer)
        print(rank,writer['rank'],local['rank'],":","db:",db_info['db']," || coll:",db_info['coll'])