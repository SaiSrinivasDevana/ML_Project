import os
import pickle
from src.exception import Custom_Exception
import sys

def save_object(filepath,obj):

    try:
        dir_name = os.path.dirname(filepath)

        os.makedirs(dir_name,exist_ok = True)

        with open(filepath,"wb") as fileobj:
            pickle.dump(obj,fileobj)
    
    except Exception as e:
        raise Custom_Exception(e,sys)
    
    
