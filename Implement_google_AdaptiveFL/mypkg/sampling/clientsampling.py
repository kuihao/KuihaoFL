import random
import numpy as np

def DynamicClientSample(rounds = 40, client_range=100, 
                        exeuting_client_number=10, 
                        sample_rate=0.1, client_id=0,
                        seed=2021):
    client_id_list = [i for i in range(client_range)]
    random.seed(seed)
    client_sample_list = np.zeros((exeuting_client_number,client_range))
    for i in range(exeuting_client_number):
        for j in range(rounds):
            client_sample_list[i,j] = random.sample(client_id_list, 1)[0]
    
    client_sample_list = client_sample_list.astype(int).tolist()
    return client_sample_list[client_id]
    
def Simulation_DynamicClientSample(
            exeuting_client_number=10,
            connect_client_number=500,
            rounds=40,
            seed=2021):
    client_id_samples_np = np.zeros([exeuting_client_number,rounds],int)

    random.seed(seed)
    for rnd in range(rounds):
        for client_i in range(exeuting_client_number):
            client_id_samples_np[client_i,rnd] = int(random.randrange(0,connect_client_number,1))

    return client_id_samples_np

def FixClientSample(client_number=10):
    '''sample rate = 10%, returns: [list of sample IDs]'''
    if client_number == 10:
        SampleID = [[6, 8, 4, 3, 0, 7, 7, 9, 1, 5,
                     4, 4, 7, 1, 2, 2, 0, 1, 8, 4,
                     0, 8, 7, 1, 3, 9, 7, 4, 8, 5, 
                     5, 6, 1, 3, 3, 7, 7, 7, 1, 8]]
    elif client_number == 100:
        SampleID = [[51, 80, 69, 35, 31, 81, 4, 56, 60, 73, 8, 40, 34, 37, 60, 9, 20, 21, 6, 95, 13, 86, 91, 70, 37, 6, 68, 87, 59, 14, 25, 77, 81, 56, 81, 35, 67, 43, 85, 47], 
                    [50, 88, 14, 28, 31, 59, 56, 60, 14, 82, 69, 28, 59, 60, 36, 17, 35, 0, 2, 68, 96, 29, 24, 96, 21, 31, 51, 54, 74, 83, 47, 88, 33, 6, 35, 9, 10, 22, 74, 57], 
                    [89, 28, 74, 17, 80, 54, 50, 99, 34, 55, 22, 80, 54, 17, 78, 88, 60, 36, 13, 44, 1, 57, 71, 37, 45, 80, 14, 9, 64, 11, 55, 41, 74, 12, 92, 12, 20, 50, 20, 46], 
                    [26, 17, 10, 14, 3, 43, 94, 43, 69, 5, 71, 96, 35, 92, 3, 70, 74, 94, 87, 53, 22, 94, 14, 4, 0, 53, 43, 4, 76, 51, 65, 61, 45, 69, 30, 32, 76, 65, 34, 42], 
                    [23, 55, 75, 44, 67, 48, 21, 68, 41, 8, 21, 29, 3, 77, 62, 64, 19, 74, 58, 84, 28, 43, 82, 70, 71, 20, 6, 43, 65, 23, 36, 23, 90, 45, 26, 88, 1, 47, 35, 7], 
                    [25, 52, 73, 10, 38, 21, 52, 65, 61, 7, 28, 99, 83, 40, 92, 95, 14, 58, 88, 19, 34, 42, 14, 52, 34, 94, 96, 28, 86, 75, 91, 82, 67, 5, 85, 38, 59, 99, 63, 36], 
                    [57, 51, 59, 37, 55, 90, 85, 67, 95, 68, 49, 1, 69, 17, 51, 68, 90, 32, 38, 46, 32, 34, 76, 29, 7, 34, 95, 97, 72, 32, 48, 13, 99, 12, 95, 2, 97, 80, 30, 51], 
                    [65, 35, 94, 82, 20, 57, 73, 74, 71, 29, 10, 81, 16, 35, 96, 72, 34, 60, 99, 86, 71, 35, 98, 52, 61, 99, 89, 75, 14, 64, 92, 97, 18, 17, 21, 42, 42, 22, 94, 27], 
                    [58, 44, 59, 94, 5, 16, 68, 19, 25, 46, 87, 46, 19, 92, 60, 28, 59, 16, 17, 2, 98, 58, 61, 22, 0, 0, 28, 56, 18, 64, 49, 50, 46, 81, 79, 83, 56, 6, 87, 10], 
                    [19, 24, 74, 29, 87, 2, 44, 7, 79, 26, 42, 17, 38, 91, 93, 30, 73, 38, 61, 49, 26, 16, 55, 99, 80, 65, 0, 64, 77, 5, 57, 16, 79, 30, 63, 61, 65, 38, 17, 31]]
    else:
        SampleID = [[]]
    return SampleID