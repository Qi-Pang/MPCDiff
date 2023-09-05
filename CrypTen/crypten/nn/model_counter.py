import pickle


# HACK: used for counting
def register_counter(args):
    global sigmoid_1_index_high
    global sigmoid_1_index_low
    global sigmoid_2_index_high
    global sigmoid_2_index_low
    global sigmoid_3_index_high
    global sigmoid_3_index_low
    global sigmoid_4_index_high
    global sigmoid_4_index_low

    global batchnorm_3_index_high
    global batchnorm_3_index_low
    global batchnorm_4_index_high
    global batchnorm_4_index_low

    global computation_time

    global batchnorm_3_shape
    global batchnorm_4_shape
    global sigmoid_1_shape
    global sigmoid_2_shape
    global sigmoid_3_shape
    global sigmoid_4_shape
    global global_repair_flag


    global_repair_flag = False

    sigmoid_1_shape = 4704
    sigmoid_2_shape = 1600
    batchnorm_3_shape = 120
    batchnorm_4_shape = 84
    if global_repair_flag:
        if args.dataset == 'MNIST':
            with open ('./results/LeNet/layer1_max.pkl', 'rb') as fp:
                sigmoid_1_index_high = pickle.load(fp)
            with open ('./results/LeNet/layer1_min.pkl', 'rb') as fp:
                sigmoid_1_index_low = pickle.load(fp)
            with open ('./results/LeNet/layer2_max.pkl', 'rb') as fp:
                sigmoid_2_index_high = pickle.load(fp)
            with open ('./results/LeNet/layer2_min.pkl', 'rb') as fp:
                sigmoid_2_index_low = pickle.load(fp)

            # # HACK: MNIST
            with open ('./results/LeNet/layer3_max.pkl', 'rb') as fp:
                batchnorm_3_index_high = pickle.load(fp)
            with open ('./results/LeNet/layer3_min.pkl', 'rb') as fp:
                batchnorm_3_index_low = pickle.load(fp)

            with open ('./results/LeNet/layer4_max.pkl', 'rb') as fp:
                batchnorm_4_index_high = pickle.load(fp)
            with open ('./results/LeNet/layer4_min.pkl', 'rb') as fp:
                batchnorm_4_index_low = pickle.load(fp)

        elif args.dataset == 'Credit':
            # HACK: Credit
            with open ('./results/Credit/layer1_max.pkl', 'rb') as fp:
                batchnorm_3_index_high = pickle.load(fp)
            with open ('./results/Credit/layer1_min.pkl', 'rb') as fp:
                batchnorm_3_index_low = pickle.load(fp)
        
        elif args.dataset == 'Bank':
            with open ('./results/Bank/layer1_max.pkl', 'rb') as fp:
                batchnorm_3_index_high = pickle.load(fp)
            with open ('./results/Bank/layer1_min.pkl', 'rb') as fp:
                batchnorm_3_index_low = pickle.load(fp)
            batchnorm_3_shape = 250
    else:
        sigmoid_1_index_high = []
        sigmoid_1_index_low = []
        sigmoid_2_index_high = []
        sigmoid_2_index_low = []
        batchnorm_3_index_high = []
        batchnorm_3_index_low = []
        batchnorm_4_index_high = []
        batchnorm_4_index_low = []


    computation_time = 0.0

    sigmoid_3_shape = batchnorm_3_shape
    sigmoid_4_shape = batchnorm_4_shape

    sigmoid_3_index_high = batchnorm_3_index_high
    sigmoid_3_index_low = batchnorm_3_index_low
    sigmoid_4_index_high = batchnorm_4_index_high
    sigmoid_4_index_low = batchnorm_4_index_low