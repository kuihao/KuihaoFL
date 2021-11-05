from datetime import datetime
import argparse

def ServerArg():
    '''To get the learning setting info. via commandline args'''
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("-m", "--mode", 
                        type=int, choices=range(0, 3), required=True,
                        help="\"0\" to start a new trainig,\n"
                             "\"1\" to recovery this model from prior record,\n"\
                             "\"2\" to load other checkpoints (weights).\n")
    parser.add_argument("-n", "--name", 
                        type=str, required=False,
                        help="Give model a unique name.")
    parser.add_argument("-pp", "--prior_path", 
                        type=str, required=False,
                        help="If model = 1, input the prior model storage file path.")
    parser.add_argument("-cp", "--checkpoint_path", 
                        type=str, required=False,
                        help="If model = 2, input the other checkpoint storage file path.")
    parser.add_argument("--cpu",
                    action="store_true",  # 引數儲存為 boolean
                    help="使否只使用CPU? 此為引數為Flag，後方不能再輸入其他內容")
    args = parser.parse_args()

    if args.mode == 1 and args.prior_path is None:
        raise Exception("If model = 1, --prior_path (-pp) is required.")
    elif args.mode == 2 and args.checkpoint_path is None:
        raise Exception("If model = 2, --checkpoint_path (-cp) is required.")
    return args

def ModelNameGenerator(UniqueName=None):
    '''Welcome prompt and generate the model name with timestemp'''
    print("  ___ _   _               _                      _   _  __       _       _                           _       _     _       ")
    print(" |_ _| |_( )___    __ _  | |__   ___  __ _ _   _| |_(_)/ _|_   _| |   __| | __ _ _   _    ___  _   _| |_ ___(_) __| | ___  ")
    print("  | || __|// __|  / _` | | '_ \ / _ \/ _` | | | | __| | |_| | | | |  / _` |/ _` | | | |  / _ \| | | | __/ __| |/ _` |/ _ \ ")
    print("  | || |_  \__ \ | (_| | | |_) |  __/ (_| | |_| | |_| |  _| |_| | | | (_| | (_| | |_| | | (_) | |_| | |_\__ \ | (_| |  __/ ")
    print(" |___|\__| |___/  \__,_| |_.__/ \___|\__,_|\__,_|\__|_|_|  \__,_|_|  \__,_|\__,_|\__, |  \___/ \__,_|\__|___/_|\__,_|\___| ")
    print("                                                                                 |___/                                     ")
    print("-------------")
    now_time = datetime.now() # current date and time
    time_str = now_time.strftime("%m_%d_%Y__%H_%M_%S")
    '' if UniqueName is None else UniqueName
    model_name = time_str+"_"+UniqueName
    return model_name