import os
import argparse
import random
import torch.utils.data
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from torch import optim
from Statistical_method import *
import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
from dataloader.Data_loader import DatasetGenerator
from train_cls import train_step_cls, train_step_cls_prox
from communication_method import communication
from models.ResNet_withoutBN import resnet50
from models.ResNet_Pre import Resnet2d, Resnet2d50


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-FL mode', '--mode', type=str, default="FedProx", help="FedProx, FedBN, siloBN, FedAvg")
parser.add_argument('-dataset_n', '--dataset_name', type=str, default="BCa_d",
                    help="selected by BCa, BCa_d, mnist, cifar10")
# num classes
parser.add_argument('-class', '--num_class', type=int, default=2, help="number of the classes")
# record data
parser.add_argument('-rd', '--record_data', type=bool, default=True, help="whether record result")
# GPU id selected
parser.add_argument('-g', '--gpu', type=str, default='1', help='gpu id to use(e.g. 0,1)')
# 客户端的数量(这里解释一下nc=5的原因，只有4个中心，4个中心的训练集合并构成了client_0，当成集中式训练)
parser.add_argument('-nc', '--num_of_clients', type=int, default=5, help='number of the clients')
# 是否采用联邦学习方式进行训练
parser.add_argument('-random_t', '--whether_random_training', type=bool, default=True,
                    help='whether random selected clients for training')
"指定编号客户端独立参与"
parser.add_argument('-idc', '--id_of_clients', type=int, default=4, help='id of the client, only can use 0,1,2,3,4')
# 随机挑选的客户端的数量
parser.add_argument('-cf', '--cfraction', type=float, default=0.4,
                    help='C fraction, 0 means 1 client, 1 means total clients')
# 训练次数(客户端更新次数)
parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')
# batchsize大小
parser.add_argument('-B', '--batchsize', type=int, default=24, help='local train batch size')
# 模型名称
parser.add_argument('-mn', '--model_name', type=str, default='ResNet_BCa', help='the model to train')
# clients selected
parser.add_argument('-cs', '--spec_clientdata', type=bool, default=False, help='the way to allocate data to clients')
# 学习率
parser.add_argument('-lr', "--learning_rate", type=float, default=0.00001, help="learning rate, \
                    use value from origin paper as default")
# parser.add_argument('-dataset', "--dataset", type=str, default="mnist", help="需要训练的数据集")
# 模型验证频率（通信频率）
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=100, help='global model save frequency(of communication)')
# n um_comm 表示通信次数，此处设置为1k
parser.add_argument('-ncomm', '--num_comm', type=int, default=500, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str,
                    default='/home/user14/sharedata/newnas/ZChang/BCA_FL_results/FedBN/',
                    help='the saving path of checkpoints')
parser.add_argument('--seed', type=int, default=57, help="固定随机数，使实验可重复")
parser.add_argument('--lr_warm_epoch', type=int, default=20)  # warmup的epoch数,一般就是10~20,为0或False则不使用
parser.add_argument('--lr_cos_epoch', type=int,
                    default=580)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用


def print_result(validation_auc, validation_Loss, valid_ID, valid_label, valid_score, val_precision, val_recall, val_f1,
                 count):
    print("validation_auc_{} :{}".format(count, validation_auc))
    print("validation_Loss_{}: {}".format(count, validation_Loss))
    print("valid_ID_{}:{}".format(count, valid_ID))
    print("valid_label_{}:{}".format(count, valid_label))
    print("valid_score_{}: {}".format(count, valid_score))
    print("val_precision_{}: {}".format(count, val_precision))
    print("val_recall_{}: {}".format(count, val_recall))
    print("val_f1_{}: {}".format(count, val_f1))


def save_folder_mk(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, "checkpoints"))


def record_and_save(preds_list, labels_list):
    preds_array = np.array(preds_list)
    labels_array = np.squeeze(np.array(labels_list))
    AUC_value = metrics.roc_auc_score(labels_array, preds_array)
    print("AUC:{}".format(AUC_value))
    ACC_value = get_accuracy(pred_value=preds_array, label_value=labels_array)
    print("ACC:{}".format(ACC_value))
    SEN_value = get_sensitivity(pred_value=preds_array, label_value=labels_array)
    print("SEN:{}".format(SEN_value))
    SPE_value = get_specificity(pred_value=preds_array, label_value=labels_array)
    print("SPE:{}".format(SPE_value))
    PREC_value = get_precision(pred_value=preds_array, label_value=labels_array)
    print("PRE:{}".format(PREC_value) + "\n")
    Threshold_value = get_best_threshold(pred_value=preds_array, label_value=labels_array)

    return AUC_value, ACC_value, SEN_value, SPE_value, PREC_value, Threshold_value


"main function"
if __name__ == "__main__":
    "data path"
    Center1_h5data_path = "/path/to/your/dir/BCA/Center_1/h5_data_128"
    Center1_train_excelpath = "/path/to/your/dir/BCA/center1_train.csv"
    Center1_test_excelpath = "/path/to/your/dir/BCA/center1_test.csv"

    Center2_h5data_path = "/path/to/your/dir/BCA/Center_2/h5_data_128"
    Center2_train_excelpath = "/path/to/your/dir/BCA/center2_train.csv"
    Center2_test_excelpath = "/path/to/your/dir/BCA/center2_test.csv"

    Center3_h5data_path = "/path/to/your/dir/BCA/Center_3/h5_data_128"
    Center3_train_excelpath = "/path/to/your/dir/BCA/center3_train.csv"
    Center3_test_excelpath = "/path/to/your/dir/BCA/center3_test.csv"

    Center4_h5data_path = "/path/to/your/dir/BCA/Center_4/h5_data_128"
    Center4_train_excelpath = "/path/to/your/dir/BCA/center4_train.csv"
    Center4_test_excelpath = "/path/to/your/dir/BCA/center4_test.csv"

    "Get parameter"
    args = parser.parse_args()
    args = args.__dict__
    print(args)
    "固定随机数"
    seed = args['seed']
    seed = seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # 为CPU设置随机数
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机数
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    "系统时间"
    now = datetime.datetime.now()
    print("now date and time: ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    if args["whether_random_training"]:
        note = "模式为联邦学习"
    else:
        note = "单个客户训练Client_{}".format(args["id_of_clients"])

    args["save_path"] = args["save_path"] + note + "{}".format(now.strftime("_%Y-%m-%d %H.%M.%S"))

    "Result save path"
    save_folder_mk(args['save_path'])

    "Record the setting and result"
    if args["record_data"]:
        test_txt = open(os.path.join(args["save_path"], "result_record_" + note + now.strftime("_%Y-%m-%d %H.%M.%S") +
                                     ".txt"), mode="a")
        "记录本次实验的运行命令"
        test_txt.write("python server.py -nc " + str(args["num_of_clients"]) + " -cf " + str(args["cfraction"]) + " -E "
                       + str(args["epoch"]) + " -B " + str(args["batchsize"]) + " -mn " + str(args["model_name"])
                       + " -ncomm " + str(args["num_comm"]) + " -iid " + str(args["IID"]) + " -lr "
                       + str(args["learning_rate"]) + " -vf " + str(args["val_freq"]) + " -g " + str(args["gpu"]))

        "记录本次实验的部分参数"
        test_txt.write(
            "dataset_name： " + str(args["dataset_name"]) + "；是否为联邦学习模式： " + str(
                args["whether_random_training"])
            + "；客户端N单独训练： " + str(args["id_of_clients"]) + "；联邦学习每轮参与聚合的客户端比例： " + str(
                args["cfraction"])
            + "；局部每轮训练epoch： " + str(args["epoch"]) + "；batchsize： " + str(args["batchsize"]) + "；学习率： " + str(
                args["learning_rate"]))
        test_txt.write("模型训练多少次保存一次： " + str(args["save_freq"]) + "；保存路径： " + str(args["save_path"])
                       + "；数据分布模式： " + str(args["IID"]))

        "记录本次实验的参数设置"
        test_txt.write("\n" + "*" * 10 + "parameter setting" + "*" * 10 + "\n")
        test_txt.write("dataset_name: {}".format(args["dataset_name"]) + "\n")
        test_txt.write("gpu_idx: {}".format(args["gpu"]) + "\n")
        test_txt.write("num_of_clients: {}".format(args["num_of_clients"]) + "\n")
        test_txt.write("cfraction: {}".format(args["cfraction"]) + "\n")
        test_txt.write("epoch: {}".format(args["epoch"]) + "\n")
        test_txt.write("batchsize: {}".format(args["batchsize"]) + "\n")
        test_txt.write("model_name: {}".format(args["model_name"]) + "\n")
        test_txt.write("learning_rate: {}".format(args["learning_rate"]) + "\n")
        test_txt.write("num_comm: {}".format(args["num_comm"]) + "\n")
        test_txt.write("save_path: {}".format(args["save_path"]) + "\n")
        test_txt.write("IID: {}".format(args["IID"]) + "\n")
        test_txt.write("\n" + "*" * 10 + "baseinfo for data" + "*" * 10 + "\n")

    "GPU setting"
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    "Initial model"
    net = None

    if args["model_name"] == "Resnet2d":
        net = Resnet2d50(in_channel=3, label_category_dict=dict(label=2), dim=2)
    if args["model_name"] == "Resnet2d_no_BN":
        net = resnet50()

    "如果有多个GPU"
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    "将Tenor 张量 放在 GPU上"
    net = net.to(dev)

    "定义损失函数"
    loss_func = F.cross_entropy

    "优化算法的，随机梯度下降法, 使用Adam下降法"
    learning_rate = args['learning_rate']
    opti = optim.Adam(net.parameters(), lr=learning_rate)

    "*********************************************打包训练数据*********************************************************"
    Center_train_loader = {}

    Center1_data_train = DatasetGenerator(path=Center1_h5data_path, excelpath=Center1_train_excelpath, Aug=True,
                                          n_class=args["num_class"],
                                          set_name='train')
    Center1_train_loader = torch.utils.data.DataLoader(dataset=Center1_data_train, batch_size=args["batchsize"],
                                                       shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    Center2_data_train = DatasetGenerator(path=Center2_h5data_path, excelpath=Center2_train_excelpath, Aug=True,
                                          n_class=args["num_class"],
                                          set_name='train')
    Center2_train_loader = torch.utils.data.DataLoader(dataset=Center2_data_train, batch_size=args["batchsize"],
                                                       shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    Center3_data_train = DatasetGenerator(path=Center3_h5data_path, excelpath=Center3_train_excelpath, Aug=True,
                                          n_class=args["num_class"],
                                          set_name='train')
    Center3_train_loader = torch.utils.data.DataLoader(dataset=Center3_data_train, batch_size=args["batchsize"],
                                                       shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    Center4_data_train = DatasetGenerator(path=Center4_h5data_path, excelpath=Center4_train_excelpath, Aug=True,
                                          n_class=args["num_class"],
                                          set_name='train')
    Center4_train_loader = torch.utils.data.DataLoader(dataset=Center4_data_train, batch_size=args["batchsize"],
                                                       shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    Center0_data_train = Center1_data_train + Center2_data_train + Center3_data_train + Center4_data_train
    Center0_train_loader = torch.utils.data.DataLoader(dataset=Center0_data_train, batch_size=args["batchsize"],
                                                       shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    #  合成center0
    Center_train_loader["client0"] = Center0_train_loader
    Center_train_loader["client1"] = Center1_train_loader
    Center_train_loader["client2"] = Center2_train_loader
    Center_train_loader["client3"] = Center3_train_loader
    Center_train_loader["client4"] = Center4_train_loader

    "**********************************************打包测试数据*****************************************************"
    Center_test_loader = {}

    Center1_data_test = DatasetGenerator(path=Center1_h5data_path, excelpath=Center1_test_excelpath, Aug=False,
                                         n_class=args["num_class"],
                                         set_name='test')
    Center1_test_loader = torch.utils.data.DataLoader(dataset=Center1_data_test, batch_size=args["batchsize"],
                                                      shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    Center2_data_test = DatasetGenerator(path=Center2_h5data_path, excelpath=Center2_test_excelpath, Aug=False,
                                         n_class=args["num_class"],
                                         set_name='test')
    Center2_test_loader = torch.utils.data.DataLoader(dataset=Center2_data_test, batch_size=args["batchsize"],
                                                      shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    Center3_data_test = DatasetGenerator(path=Center3_h5data_path, excelpath=Center3_test_excelpath, Aug=False,
                                         n_class=args["num_class"],
                                         set_name='test')
    Center3_test_loader = torch.utils.data.DataLoader(dataset=Center3_data_test, batch_size=args["batchsize"],
                                                      shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    Center4_data_test = DatasetGenerator(path=Center4_h5data_path, excelpath=Center4_test_excelpath, Aug=False,
                                         n_class=args["num_class"],
                                         set_name='test')
    Center4_test_loader = torch.utils.data.DataLoader(dataset=Center4_data_test, batch_size=args["batchsize"],
                                                      shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    Center_test_loader["client1"] = Center1_test_loader
    Center_test_loader["client2"] = Center2_test_loader
    Center_test_loader["client3"] = Center3_test_loader
    Center_test_loader["client4"] = Center4_test_loader

    # ---------------------------------------以上准备工作已经完成------------------------------------------#
    "客户端分配模式"
    if args["spec_clientdata"]:
        "当num_in_comm = 1时，是某个中心单独训练，然后在测试集上进行测试"
        "当num_in_comm = client_num时，是所有中心联合训练，并且整合参数，然后在测试集上进行测试"
        num_in_comm = 1  # int(args['num_of_clients'] * args['cfraction'])
    else:
        "每次随机选取args['num_of_clients'] * args['cfraction']个Clients"
        num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    print("num_in_comm:{}".format(num_in_comm))

    "sever model"
    sever_model = net
    "client models"
    models = {}
    optimizers = {}
    client_weights = [1 / num_in_comm for i in range(num_in_comm)]

    Test_all_AUC_list = []

    Test_center_AUC_list, Test_center1_AUC_list, Test_center2_AUC_list, Test_center3_AUC_list, Test_center4_AUC_list = [], [], [], [], []
    Test_center_AUC_list.append(Test_center1_AUC_list)
    Test_center_AUC_list.append(Test_center2_AUC_list)
    Test_center_AUC_list.append(Test_center3_AUC_list)
    Test_center_AUC_list.append(Test_center4_AUC_list)

    train_loss_list = []
    train_lr_list = []

    Test_client_list = ["client1", "client2", "client3", "client4"]

    test_all_record = np.array(
        ["Epoch", "ACC", "SEN", "SPEC", "PREC", "AUC"])
    test_center1_record = np.array(
        ["Epoch", "ACC", "SEN", "SPEC", "PREC", "Threshold", "AUC"])
    test_center2_record = np.array(
        ["Epoch", "ACC", "SEN", "SPEC", "PREC", "Threshold", "AUC"])
    test_center3_record = np.array(
        ["Epoch", "ACC", "SEN", "SPEC", "PREC", "Threshold", "AUC"])
    test_center4_record = np.array(
        ["Epoch", "ACC", "SEN", "SPEC", "PREC", "Threshold", "AUC"])

    auc_record = np.array(["Test_all", "Test_center1", "Test_center2", "Test_center3", "Test_center4"])

    test_center_list = []
    test_center_list.append(test_center1_record)
    test_center_list.append(test_center2_record)
    test_center_list.append(test_center3_record)
    test_center_list.append(test_center4_record)

    Max_AUC = 0

    "num_comm 表示通信次数，即本地和客户端交换模型的次数"
    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))

        "对随机选的将num_of_clients个客户端进行随机排序"
        order = np.random.permutation(range(1, args['num_of_clients']))
        print("client_total_num:{},  client_order:{}:".format(len(order), order))
        "生成个客户端"
        if args["whether_random_training"]:
            clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
            print("客户端" + str(clients_in_comm))
            for idx in clients_in_comm:
                models[idx] = net
            for idx in clients_in_comm:
                optimizers[idx] = optim.Adam(params=models[idx].parameters(), lr=learning_rate)
            "每一轮训练的epoch数"
            for E in range(args["epoch"]):
                print(
                    "=============================Train Epoch communication{}_local epoch{}=========================".format(
                        i, E))
                # 每个Client基于当前模型参数和自己的数据训练并更新模型
                # 返回每个Client更新后的参数
                # 这里的clients
                for k, client in enumerate(tqdm(clients_in_comm)):
                    # 获取当前Client训练得到的参数
                    # 这一行代码表示Client端的训练函数，我们详细展开：
                    # local_parameters 得到客户端的局部变量
                    opti_client = optimizers[client]
                    if args["mode"].lower() == 'fedprox':
                        if i > 0:
                            current_lr, loss_value = train_step_cls_prox(
                                train_loader=Center_train_loader[client],
                                model=models[client], epoch=i, optimizer=opti_client,
                                criterion=loss_func, args=args,
                                sever_model=sever_model,
                            )
                        else:
                            current_lr, loss_value = train_step_cls(train_loader=Center_train_loader[client],
                                                                    model=net, epoch=i, optimizer=opti,
                                                                    criterion=loss_func, args=args,
                                                                    global_parameters=sever_model.state_dict()
                                                                    )

                    else:
                        current_lr, loss_value = train_step_cls(train_loader=Center_train_loader[client],
                                                                model=net, epoch=i, optimizer=opti,
                                                                criterion=loss_func, args=args,
                                                                global_parameters=sever_model.state_dict()
                                                                )

            train_loss_list.append(loss_value)
            train_lr_list.append(current_lr)
            server_model, models = communication(args["mode"], sever_model, models, client_weights, clients_in_comm)
        else:
            clients_in_comm = ['client' + str(args["id_of_clients"])]
            print("客户端" + str(clients_in_comm))
            for k, client in enumerate(clients_in_comm):
                models[clients_in_comm[0]] = net
                optimizers[clients_in_comm[0]] = optim.Adam(params=models[clients_in_comm[0]].parameters(), lr=learning_rate)
                current_lr, loss_value = train_step_cls(train_loader=Center_train_loader[client],
                                                    model=net, epoch=i, optimizer=opti,
                                                    criterion=loss_func, args=args,
                                                    global_parameters=sever_model.state_dict()
                                                    )
                server_model = net
                for key in server_model.state_dict().keys():
                    models[client].state_dict()[key].data.copy_(server_model.state_dict()[key])
        assert args["num_class"] >= 2, "num_class have to greater than 1"
        net = server_model
        sum_accu = 0
        num = 0
        All_test_preds_list = []
        All_test_labels_list = []

        Test_all_Client_AUC_list = []
        Start_test = True

        if Start_test:
            all_index_slice_list = []
            id_list = []
            Client_true_list = []
            Client_sum_list = []
            same_commo_AUC = 0
            same_commo_ACC = 0
            same_commo_SEN = 0
            same_commo_SPC = 0
            same_commo_PREC = 0
            for count_client, client_i_test in enumerate(Test_client_list):
                Client_test_preds_list = []
                Client_test_labels_list = []
                Client_index_slice_list = []
                for num, (image, mask, label, sign) in enumerate(Center_test_loader[client_i_test]):
                    torch.set_grad_enabled(False)
                    "读取初始数据"
                    image = image.to(dev)
                    output = net(image)
                    output = list(output.values())[0]
                    pred_label = output
                    "开始统计指标"
                    for count, index_slice in enumerate(sign):
                        "统计分类指标"
                        Client_test_preds_list.append(np.array(pred_label.cpu().detach().numpy()[:, 1])[count])
                        Client_test_labels_list.append(np.squeeze(label.cpu().detach().numpy().astype(int))[count])

                        "统计切片id"
                        Client_index_slice_list.append(int(index_slice.cpu().detach().numpy()))
                        all_index_slice_list.append(int(index_slice.cpu().detach().numpy()))

                Client_preds_mean_list = []
                Client_labels_mean_list = []

                id_list_simple = list(set(Client_index_slice_list))

                for id_c in id_list_simple:
                    "找到相关索引"
                    index_same = np.where(np.array(Client_index_slice_list) == id_c)

                    "***************************tumour级别的分类预测值和label计算***************************"
                    tumour_preds_value_mean = np.sum(np.array(Client_test_preds_list)[index_same]) / len(
                        np.array(Client_test_preds_list)[index_same])
                    tumour_label_value_mean = int(np.sum(np.array(Client_test_labels_list)[index_same]) / len(
                        np.array(Client_test_labels_list)[index_same]))
                    Client_preds_mean_list.append(tumour_preds_value_mean)
                    Client_labels_mean_list.append(tumour_label_value_mean)

                "**********************************Client级别的统计分类预测指标**************************************"
                print("*" * 10 + "AUC for client{}".format(count_client + 1) + "*" * 10)
                AUC_value, ACC_value, SEN_value, SPE_value, PREC_value, Thredshold_value = record_and_save(
                    Client_preds_mean_list,
                    Client_labels_mean_list)
                Test_center_AUC_list[count_client].append(AUC_value)
                same_commo_AUC += AUC_value
                same_commo_ACC += ACC_value
                same_commo_SEN += SEN_value
                same_commo_SPC += SPE_value
                same_commo_PREC += PREC_value

                test_center_list[count_client] = np.vstack((test_center_list[count_client], np.array(
                    [i + 1, format(ACC_value, ".3f"), format(SEN_value, ".3f"), format(SPE_value, ".3f"),
                     format(PREC_value, ".3f"), format(Thredshold_value, ".3f"),
                     format(AUC_value, ".3f")])))

                # 保存各预测值
                pred_value_title = np.array(["id", "pred", "label"])
                pred_value_list = np.zeros([len(id_list_simple), 3])
                pred_value_list[:, 0] = np.array(id_list_simple).T
                pred_value_list[:, 1] = np.array(Client_preds_mean_list).T
                pred_value_list[:, 2] = np.array(Client_labels_mean_list).T
                pred_value_title = np.vstack((pred_value_title, pred_value_list))

                excel_save_path = os.path.join(args["save_path"], 'prediction',
                                               client_i_test + '_' + str(i + 1) + "_" + 'pre.xlsx')
                if not os.path.exists(os.path.join(args["save_path"], 'prediction')):
                    os.makedirs(os.path.join(args["save_path"], 'prediction'))
                writer = pd.ExcelWriter(excel_save_path)
                pred_value_title = pd.DataFrame(pred_value_title)
                pred_value_title.to_excel(writer, 'pred_value', float_format='%.5f')
                writer.close()

            "*******************************所有测试集一起的统计分类预测指标******************************************"
            print("*" * 10 + "Result for all_test_client:" + "*" * 10)

            ACC_value_avg = same_commo_ACC / 4
            SEN_value_avg = same_commo_SEN / 4
            SPE_value_avg = same_commo_SPC / 4
            AUC_value_avg = same_commo_AUC / 4
            Test_all_AUC_list.append(AUC_value_avg)
            PREC_value_avg = same_commo_PREC / 4
            test_all_record = np.vstack((test_all_record, np.array(
                [i + 1, format(ACC_value_avg, ".3f"), format(SEN_value_avg, ".3f"), format(SPE_value_avg, ".3f"),
                 format(PREC_value_avg, ".3f"),
                 format(AUC_value_avg, ".3f")])))

            # 保存各预测值
            # pred_value_title = np.array(["pred", "label"])
            # pred_value_list = np.zeros([len(All_test_labels_list), 2])
            # pred_value_list[:, 0] = np.array(All_test_preds_list).T
            # pred_value_list[:, 1] = np.array(All_test_labels_list).T
            # pred_value_title = np.vstack((pred_value_title, pred_value_list))

            # excel_save_path = os.path.join(args["save_path"], 'prediction',
            #                                "Test_all" + '_' + str(i + 1) + "_" + 'pre.xlsx')
            # if not os.path.exists(os.path.join(args["save_path"], 'prediction')):
            #     os.makedirs(os.path.join(args["save_path"], 'prediction'))
            # writer = pd.ExcelWriter(excel_save_path)
            # pred_value_title = pd.DataFrame(pred_value_title)
            # pred_value_title.to_excel(writer, 'pred_value', float_format='%.3f')
            # writer.close()

            if args["record_data"]:
                if args["num_class"] == 2:

                    test_txt.write("\n" + "communicate round " + str(i + 1) + "  ")
                    test_txt.write("loss_value: " + str(float(loss_value)) + "\n")

                    "保存分类结果"
                    test_txt.write("*" * 10 + "AUC result for all test data" + "*" * 10 + "\n" +
                                   "AUC: " + str(AUC_value_avg) + "\n" +
                                   'ACC: ' + str(ACC_value_avg) + "\n" +
                                   "SEN: " + str(SEN_value_avg) + "\n" +
                                   "SPE: " + str(SPE_value_avg) + "\n" +
                                   "PRE: " + str(PREC_value_avg) + "\n")

                    test_txt.write("*" * 10 + "AUC result for test data for Center1" + "*" * 10 + "\n" +
                                   "AUC: " + str(test_center_list[0][i + 1][-1]) + "\n" +
                                   'ACC: ' + str(test_center_list[0][i + 1][1]) + "\n" +
                                   "SEN: " + str(test_center_list[0][i + 1][2]) + "\n" +
                                   "SPE: " + str(test_center_list[0][i + 1][3]) + "\n" +
                                   "PRE: " + str(test_center_list[0][i + 1][4]) + "\n")

                    test_txt.write("*" * 10 + "AUC result for test data for Center2" + "*" * 10 + "\n" +
                                   "AUC: " + str(test_center_list[1][i + 1][-1]) + "\n" +
                                   'ACC: ' + str(test_center_list[1][i + 1][1]) + "\n" +
                                   "SEN: " + str(test_center_list[1][i + 1][2]) + "\n" +
                                   "SPE: " + str(test_center_list[1][i + 1][3]) + "\n" +
                                   "PRE: " + str(test_center_list[1][i + 1][4]) + "\n")

                    test_txt.write("*" * 10 + "AUC result for test data for Center3" + "*" * 10 + "\n" +
                                   "AUC: " + str(test_center_list[2][i + 1][-1]) + "\n" +
                                   'ACC: ' + str(test_center_list[2][i + 1][1]) + "\n" +
                                   "SEN: " + str(test_center_list[2][i + 1][2]) + "\n" +
                                   "SPE: " + str(test_center_list[2][i + 1][3]) + "\n" +
                                   "PRE: " + str(test_center_list[2][i + 1][4]) + "\n")

                    test_txt.write("*" * 10 + "AUC result for test data for Center4" + "*" * 10 + "\n" +
                                   "AUC: " + str(test_center_list[3][i + 1][-1]) + "\n" +
                                   'ACC: ' + str(test_center_list[3][i + 1][1]) + "\n" +
                                   "SEN: " + str(test_center_list[3][i + 1][2]) + "\n" +
                                   "SPE: " + str(test_center_list[3][i + 1][3]) + "\n" +
                                   "PRE: " + str(test_center_list[3][i + 1][4]) + "\n")

                else:
                    test_txt.write("communicate round " + str(i + 1) + "  ")
                    test_txt.write("loss_value: " + str(float(loss_value)) + "  ")
                    test_txt.write('accuracy: ' + str(float(sum_accu / num)) + "\n")

            "保存每个comm的指标和预测值"
            excel_save_path = os.path.join(args['save_path'], 'record.xlsx')
            record = pd.ExcelWriter(excel_save_path)
            for count_c in range(len(test_center_list)):
                test_center_list[count_c] = pd.DataFrame(test_center_list[count_c])
                test_center_list[count_c].to_excel(record, "client_{}".format(count_c + 1),
                                                   float_format='%.5f')

            auc_record = np.vstack((np.array(Test_all_AUC_list).T, np.array(Test_center_AUC_list[0]).T,
                                    np.array(Test_center_AUC_list[1]).T,
                                    np.array(Test_center_AUC_list[2]).T,
                                    np.array(Test_center_AUC_list[3]).T))

            auc_record = pd.DataFrame(auc_record)
            auc_record.to_excel(record, 'auc_all', float_format='%.5f')
            record.close()

            "save model"
            if AUC_value > Max_AUC:
                Max_AUC = AUC_value
                "save model"
                torch.save(net, os.path.join(os.path.join(args['save_path'], "checkpoints"),
                                             'comm{}_AUC{}'.format(i + 1, Max_AUC) + '.pth'))

    if args["record_data"]:
        All_Test_Max_AUC = max(Test_all_AUC_list)
        Max_AUC_index = Test_all_AUC_list.index(max(Test_all_AUC_list))
        test_txt.write("在所有测试数据最大AUC值为: " + str(All_Test_Max_AUC) + "， 在通讯轮次:{}".format(
            str(Max_AUC_index + 1)) + "\n")

        for index, auc_list in enumerate(Test_center_AUC_list):
            if auc_list != []:
                test_max_auc = max(auc_list)
                max_AUC_index = auc_list.index(max(auc_list))
                test_txt.write(
                    "在Center_{}测试数据最大AUC值为: ".format(index + 1) + str(test_max_auc) + "， 在通讯轮次:{}".format(
                        str(max_AUC_index)))
                plt.figure()
                plt.plot(range(len(auc_list)), auc_list)
                plt.title("ROC Curve")
                plt.xlabel("Comm_order")
                plt.ylabel("AUC")
                plt.savefig(os.path.join(args["save_path"], "Test_center{}_AUC_curve.png".format(index + 1)))
                plt.close()
        test_txt.close()

    "画图"
    plt.figure()
    plt.plot(range(len(Test_all_AUC_list)), Test_all_AUC_list)
    plt.title("ROC Curve")
    plt.xlabel("Comm_order")
    plt.ylabel("AUC")
    # plt.legend(loc='lower right')
    plt.savefig(os.path.join(args["save_path"], "Test_all_AUC_curve.png"))
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.title("Train Loss Curve")
    plt.xlabel("training epoch")
    plt.ylabel("Loss value")
    plt.savefig(os.path.join(args["save_path"], "Loss_curve.png"))
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(range(len(train_lr_list)), train_lr_list)
    plt.title("Train Learning Rate Curve")
    plt.xlabel("training epoch")
    plt.ylabel("Learning Rate value")
    plt.savefig(os.path.join(args["save_path"], "Learning_Rate_curve.png"))
    plt.show()
    plt.close()
