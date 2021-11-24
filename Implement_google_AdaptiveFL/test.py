'''
from mypkg import mylog, secure_mkdir

model_name = 'test'
log_folder = secure_mkdir("FL_log"+"/"+model_name)

log_text = f'*** FL Traing Record ***\n' \
           f'77777'

mylog(log_text,log_folder+'/log')'''
vari, vari_name = 123, '2222222'
print(vari, vari_name)