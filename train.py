import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from tensorboardX import SummaryWriter
import argparse

import time
import tqdm
import random
from SK import *
from SD import *
import numpy as np
import shutil, os, copy

curtime = int(time.time())

random.seed(time.time())

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
parser.add_argument('-glr', '--glr', type=float, required=True, help="generator learning rate")
parser.add_argument('-dlr', '--dlr', type=float, required=True, help="discriminator learning rate")
parser.add_argument('--nosave', action='store_true', help="save log to tensor board")

args = parser.parse_args()


print("loading training data...")
video = torch.load("datasets/reorganized_training_dataset_video_v3.tar")
summary = torch.load("datasets/reorganized_training_dataset_summary_v3.tar")

print("loading training data ended")

print("loading test data...")
test_video = torch.load("datasets/reorganized_test_dataset_video_v3.tar")
test_summary = torch.load("datasets/reorganized_test_dataset_summary_v3.tar")
print("loading test data ended")

PATH_record = "saved_models/loss_record_3.tar"
PATH_model = "saved_models"

EPOCH = 500

if not args.nosave:
    with open("hparameters.txt", "a") as writefile:
        writefile.write("{}, {}, {}\n".format(curtime, args.glr, args.dlr))

reconstruction_error_coeff = 0.5
diversity_error_coeff = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vid_features = []
sum_features = []

def get_fscore(lfunction):
    fscore_list = [] #12개 영상에 대해서 test. 평균을 결과로 출력

    # 구해야 하는 것:
    # 1. X: 모델이 생성한 요약 프레임 리스트 (0과 1로 이루어진 것)
    # 2. Y: 실제 요약 프레임 리스트 (0과 1로 이루어진 것)
    
    # 계산할 것:
    # 1. P: (X, Y overlap되는 부분) / X 길이
    # 2. R: (X, Y overlap되는 부분) / Y 길이
    # 3. F: (2*P*R) / P+R

    for i in range(12):
        vid = test_video["feature"][i]
        vid_copy = copy.deepcopy(vid).cpu().data.numpy()
        summ = test_summary["feature"][i]
        summ_copy = copy.deepcopy(summ).cpu().data.numpy()

        vmean = torch.mean(vid)
        vstd = torch.std(vid)
        vid = (vid-vmean) / (vstd + 1e-8)

        smean = torch.mean(summ)
        sstd = torch.std(summ)
        summ = (summ-smean) / (sstd + 1e-8)

        vid = (vid + 1e-5 + 1) / 2
        summ = (summ + 1e-5 + 1) / 2

        # X값 구하기
        gen_summary,v1,v2, g_frames = S_K(vid)
        X = list(g_frames.cpu().data.numpy())


        # Y값 구하기
        vid_list  = np.transpose(vid_copy[0], (1,2,0))[0]
        sum_list  = np.transpose(summ_copy[0], (1,2,0))[0]
        # vid_list = []
        # sum_list = []


        Y = [0 for i in range(vid_list.shape[0])]

        idx = 0
        for a in vid_list:
            for b in sum_list:
                if np.array_equal(a, b):
                    Y[idx] = 1
                    continue
            
            idx += 1
            
        len_X = np.asarray(X).sum()
        len_Y = np.asarray(Y).sum()

        count_overlap = 0

        for i in range(len(X)):
            if X[i] == 1 and Y[i] == 1:
                count_overlap +=1
        P = count_overlap/ (len_X + 1e-16)
        R = count_overlap/(len_Y + 1e-16)
        F = ((2*P*R) / ((P+R) + 1e-16))
        fscore_list.append(F)

    mean_fscore = float(np.asarray(fscore_list).mean())
    print("F-score: {}".format(mean_fscore))


    fscore_save_filename = "result/{}_fscore.txt".format(curtime)
    if os.path.isfile(fscore_save_filename):
        with open("result/{}_fscore.txt".format(curtime), "a") as writefile:
            writefile.write("{}\n".format(mean_fscore))
    else:
        with open("result/{}_fscore.txt".format(curtime), "w") as writefile:
            writefile.write("{}\n".format(mean_fscore))


    # return float(np.asarray(fscore_list).mean()/len(fscore_list))    

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        #init.xavier_normal_(m.weight.data)
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        #init.xavier_normal_(m.weight.data)
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


S_K = SK().to(device)
S_D = SD().to(device)

optimizerS_K = optim.Adam(S_K.parameters(), lr=args.glr)
optimizerS_D = optim.SGD(S_D.parameters(), lr=args.dlr)

scheduler_S_K = optim.lr_scheduler.StepLR(optimizerS_K, step_size=8000, gamma=0.05)
scheduler_S_D = optim.lr_scheduler.StepLR(optimizerS_D, step_size=8000, gamma=0.05)

if not args.nosave:
    writer = SummaryWriter()

mode = 0

if mode==0:
    print("first train")

    time_list = []
    S_K_iter_loss_list = []
    reconstruct_iter_loss_list = []
    diversity_iter_loss_list = []
    S_D_real_iter_loss_list = []
    S_D_fake_iter_loss_list = []
    S_D_total_iter_loss_list = []
    fscore_list = []
    
    S_K.apply(weights_init)
    S_D.apply(weights_init)
    S_K.train()
    S_D.train()
elif mode==1:
    print("continue train")
    checkpoint_loss = torch.load(PATH_record)
    time_list = checkpoint_loss['time_list']; 

    iteration = len(time_list)-1
    PATH_model_load = "{}{}{:0>7d}{}".format(PATH_model, "/iter_", iteration, ".tar"); 
    checkpoint_model = torch.load(PATH_model_load)
    S_K.load_state_dict(checkpoint_model['S_K_state_dict'])
    S_D.load_state_dict(checkpoint_model['S_D_state_dict'])
    optimizerS_K.load_state_dict(checkpoint_model['optimizerS_K_state_dict'])
    optimizerS_D.load_state_dict(checkpoint_model['optimizerS_D_state_dict'])
    S_K.train()
    S_D.train()

    S_K_iter_loss_list = checkpoint_loss['S_K_iter_loss_list']
    reconstruct_iter_loss_list = checkpoint_loss['reconstruct_iter_loss_list']
    diversity_iter_loss_list = checkpoint_loss['diversity_iter_loss_list']
    S_D_real_iter_loss_list = checkpoint_loss['S_D_real_iter_loss_list']
    S_D_fake_iter_loss_list = checkpoint_loss['S_D_fake_iter_loss_list']
    S_D_total_iter_loss_list = checkpoint_loss['S_D_total_iter_loss_list']

    if not args.nosave:
        for idx in range(len(time_list)):
            writer.add_scalar("loss/S_K", S_K_iter_loss_list[idx], idx, time_list[idx])
            writer.add_scalar("loss/reconstruction", reconstruct_iter_loss_list[idx], idx, time_list[idx]) 
            writer.add_scalar("loss/diversity", diversity_iter_loss_list[idx], idx, time_list[idx]) 
            writer.add_scalar("loss/S_D_real", S_D_real_iter_loss_list[idx], idx, time_list[idx])
            writer.add_scalar("loss/S_D_fake", S_D_fake_iter_loss_list[idx], idx, time_list[idx])
            writer.add_scalar("loss/S_D_total", S_D_total_iter_loss_list[idx], idx, time_list[idx])
else:
    print("please select mode 0 or 1")



criterion = nn.BCELoss()
flossfuntion = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    random.shuffle(video["feature"])
    random.shuffle(summary["feature"])


    tqdm_range = tqdm.trange(len(video["feature"]))
    for i in tqdm_range:
        try:
            vd = video["feature"][i]
            sd = summary["feature"][i]

            vmean = torch.mean(vd)
            vstd = torch.std(vd)
            vd = (vd-vmean) / (vstd + 1e-8)

            smean = torch.mean(sd)
            sstd = torch.std(sd)
            sd = (sd-smean) / (sstd + 1e-8)

            vd = (vd + 1e-5 + 1) / 2
            sd = (sd + 1e-5 + 1) / 2

            # update SG #
            S_K.zero_grad()

            S_K_summary,index_mask,_,g_f = S_K(vd)



            output = S_D(S_K_summary)
            
            label = torch.full((1,), 1, device=device)

            # adv. loss
            errS_K = criterion(output, label)

            outputs_reconstruct = S_K_summary.view(1,1024,-1) # [1,1024,1,T] -> [1,1024,T] 
            mask = index_mask.view(1,1,-1) # [1,1,1,T] -> [1,1,T] 
            feature = vd.view(1,1024,-1) # [1,1024,1,T] -> [1,1024,T] 
            

            feature_select = feature*mask # [1,1024,T]
            outputs_reconstruct_select = outputs_reconstruct*mask # [1,1024,T]
            feature_diff_1 = torch.sum((feature_select-outputs_reconstruct_select)**2, dim=1) # [1,T]
            feature_diff_1 = torch.sum(feature_diff_1, dim=1) # [1]

            mask_sum = torch.sum(mask, dim=2) # [1,1]
            mask_sum = torch.sum(mask_sum, dim=1) # [1]

            reconstruct_loss = torch.mean(feature_diff_1/(mask_sum + 1e-8)) # scalar

            # diversity loss
            batch_size, feat_size, frames = outputs_reconstruct.shape # [1,1024,T]

            outputs_reconstruct_norm = torch.norm(outputs_reconstruct, p=2, dim=1, keepdim=True) # [1,1,T]

            normalized_outputs_reconstruct = outputs_reconstruct/outputs_reconstruct_norm # [1,1024,T]

            normalized_outputs_reconstruct_reshape = normalized_outputs_reconstruct.permute(0, 2, 1) # [1,T,1024]

            similarity_matrix = torch.bmm(normalized_outputs_reconstruct_reshape, normalized_outputs_reconstruct) # [1,T,T]

            mask_trans = mask.permute(0,2,1) # [1,T,1]
            mask_matrix = torch.bmm(mask_trans, mask) # [1,T,T]
            # filter out non key
            similarity_matrix_filtered = similarity_matrix*mask_matrix # [1,T,T]

            diversity_loss = 0
            acc_batch_size = 0
            for j in range(batch_size):
                batch_similarity_matrix_filtered = similarity_matrix_filtered[j,:,:] # [T,T]
                batch_mask = mask[j,:,:] # [T,T]
                if batch_mask.sum() < 2:
                    batch_diversity_loss = 0
                else:
                    batch_diversity_loss = (batch_similarity_matrix_filtered.sum()-batch_similarity_matrix_filtered.trace())/((batch_mask.sum()*(batch_mask.sum()-1)) + 1e-8)
                    acc_batch_size += 1

                diversity_loss += batch_diversity_loss

            if acc_batch_size>0:
                diversity_loss /= (acc_batch_size + 1e-8)
            else:
                diversity_loss = 0
            
            diversity_loss += 1e-8

            S_K_total_loss = errS_K*0.5 + reconstruction_error_coeff*reconstruct_loss + diversity_error_coeff*diversity_loss*0.5 + 1e-8 

            S_K_total_loss.backward()

            # update
            optimizerS_K.step()
            scheduler_S_K.step()
            
            # update S_D #

            S_D.zero_grad()

            # real summary #
            output = S_D(sd)
            label.fill_(1)
            err_S_D_real = criterion(output, label)
            err_S_D_real.backward()

            S_K_summary,idx1,idx2, selc = S_K(vd)
            
            
            output = S_D(S_K_summary.detach()); 
            label.fill_(0)
            err_S_D_fake = criterion(output, label)
            err_S_D_fake.backward()

            S_D_total_loss = err_S_D_real*0.3+err_S_D_fake*0.7 + 1e-8


            optimizerS_D.step()
            scheduler_S_D.step()
            
            # record
            time_list.append(time.time())
            S_K_iter_loss_list.append(errS_K)
            reconstruct_iter_loss_list.append(reconstruction_error_coeff*reconstruct_loss)
            diversity_iter_loss_list.append(diversity_error_coeff*diversity_loss)
            S_D_real_iter_loss_list.append(err_S_D_real)
            S_D_fake_iter_loss_list.append(err_S_D_fake)
            S_D_total_iter_loss_list.append(S_D_total_loss)

            iteration = len(time_list)-1
            
            if ((iteration+1)%(150*50)==0): # save every 50 epoch
                PATH_model_save = "{}{}{:0>7d}{}".format(PATH_model, "/iter_", iteration, "_3.tar")
                S_K_state_dict = S_K.state_dict()
                optimizerS_K_state_dict = optimizerS_K.state_dict()
                S_D_state_dict = S_D.state_dict()
                optimizerS_D_state_dict = optimizerS_D.state_dict()

                torch.save({
                        "S_K_state_dict": S_K_state_dict,
                        "optimizerS_K_state_dict": optimizerS_K_state_dict,
                        "S_D_state_dict": S_D_state_dict,
                        "optimizerS_D_state_dict":  optimizerS_D_state_dict
                        }, PATH_model_save)

                print("model is saved in {}".format(PATH_model_save))

                torch.save({
                        "S_K_iter_loss_list": S_K_iter_loss_list,
                        "reconstruct_iter_loss_list": reconstruct_iter_loss_list,
                        "diversity_iter_loss_list": diversity_iter_loss_list,
                        "S_D_real_iter_loss_list": S_D_real_iter_loss_list,
                        "S_D_fake_iter_loss_list": S_D_fake_iter_loss_list,
                        "S_D_total_iter_loss_list": S_D_total_iter_loss_list,
                        # "fscore_list": fscore_list,
                        "time_list": time_list
                        }, PATH_record)

                print("loss record is saved in {}".format(PATH_record))

                print("key frame prob", mask.view(-1))
        
        except Exception as e: 
            print("error")
            print(e)
            # exit()
            continue


        # send to tensorboard
        # if not args.nosave:
        #     writer.add_scalar("loss/S_K", S_K_iter_loss_list[iteration], iteration, time_list[iteration])   
        #     writer.add_scalar("loss/S_D_real", S_D_real_iter_loss_list[iteration], iteration, time_list[iteration])
        #     writer.add_scalar("loss/S_D_fake", S_D_fake_iter_loss_list[iteration], iteration, time_list[iteration])
        #     writer.add_scalar("loss/S_D_total", S_D_total_iter_loss_list[iteration], iteration, time_list[iteration])
        # # writer.add_scalar("loss/fscore", fscore_list[iteration], iteration, time_list[iteration])


        # # writer.add_scalar("loss/reconstruction", reconstruct_iter_loss_list[iteration], iteration, time_list[iteration]) 
        # # writer.add_scalar("loss/diversity", diversity_iter_loss_list[iteration], iteration, time_list[iteration])
    
    if (epoch+1) % 1 == 0 and not args.nosave:
    # if not args.nosave:
        get_fscore(flossfuntion)

if not args.nosave:
    writer.close()