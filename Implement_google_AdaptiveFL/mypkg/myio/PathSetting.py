import os
# os.path.join(path1,path2,file)
def secure_mkdir(forder_parh)->str:
    ''' check exist before mkdir '''
    os.makedirs(forder_parh, exist_ok=True)
    return forder_parh