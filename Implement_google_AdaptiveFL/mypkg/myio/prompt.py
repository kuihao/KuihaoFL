from datetime import datetime

def KeepPriorTrain(UniqueModelName=''):
    '''Welcome prompt: keep prior-training?'''
    print("  ___ _   _               _                      _   _  __       _       _                           _       _     _       ")
    print(" |_ _| |_( )___    __ _  | |__   ___  __ _ _   _| |_(_)/ _|_   _| |   __| | __ _ _   _    ___  _   _| |_ ___(_) __| | ___  ")
    print("  | || __|// __|  / _` | | '_ \ / _ \/ _` | | | | __| | |_| | | | |  / _` |/ _` | | | |  / _ \| | | | __/ __| |/ _` |/ _ \ ")
    print("  | || |_  \__ \ | (_| | | |_) |  __/ (_| | |_| | |_| |  _| |_| | | | (_| | (_| | |_| | | (_) | |_| | |_\__ \ | (_| |  __/ ")
    print(" |___|\__| |___/  \__,_| |_.__/ \___|\__,_|\__,_|\__|_|_|  \__,_|_|  \__,_|\__,_|\__, |  \___/ \__,_|\__|___/_|\__,_|\___| ")
    print("                                                                                 |___/                                     ")
    print("-------------")

    now_time = datetime.now() # current date and time
    time_str = now_time.strftime("%m_%d_%Y__%H_%M_%S")
  
    input_error = True # default
    model_version_name = time_str+"_"+UniqueModelName
    new_cppath = ''
    while(input_error):
        train_mode = int(input("Input:\n\"0\" to start a new trainig,\n"
        "\"1\" to recovery this model from prior record,\n"\
        "\"2\" to load other checkpoints (weights).\n"))
        if train_mode == 0:
            print("let's play a new game!")
            input_error = False
        elif train_mode == 1:
            print("Keep prior-traing going~~~")
            model_version_name = input("Prior model version name?\n")
            input_error = False
        elif train_mode == 2:
            print("What a great idea! Let's put something new to it.")
            #model_version_name = input("Prior model version name?\n")
            new_cppath = input("Where's the new checkpoint path?(forder name)\n")
            input_error = False
        else:
            input_error = True

    return train_mode, model_version_name, new_cppath

#train_mode, model_version_name = KeepPriorTrain()
#print(train_mode)
#print(model_version_name)