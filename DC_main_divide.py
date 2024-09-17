import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
### データセット分割用
from torch.utils.data import random_split  
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from torch.utils.data import Subset


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=150, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=500, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=32, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=10, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
### # 分割数を指定
    parser.add_argument('--split_num', type=int, default=500, help='number of dataset splits')  

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)

##### Get Dataset && Set up Eval pool
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    print("====================================")
    print("dst_train: ", len(dst_train))
    print("dst_test: ", len(dst_test))
    print("====================================")

    # dst_trainの各クラスのデータ数を確認
    for i in range(num_classes):
        print(f"Class {i} Length: ", len([1 for _, target in dst_train if target == i]))

    # dst_testの各クラスのデータ数を確認
    for i in range(num_classes):
        print(f"Class {i} Length: ", len([1 for _, target in dst_test if target == i]))
##### Datasetを分割
    num_splits = args.split_num
    class_indices = {i: [] for i in range(num_classes)}

    # クラスごとのインデックスを取得
    for i, (img, label) in enumerate(dst_train):
        class_indices[label].append(i)

    # 各クラスを分割数に合わせて均等に分割
    subsets = []
    for class_idx, indices in class_indices.items():
        indices = torch.tensor(indices)
        indices = indices[torch.randperm(len(indices))]
        split_size = len(indices) // num_splits

        for i in range(num_splits):
            subset_indices = indices[i * split_size: (i + 1) * split_size]
            if i >= len(subsets):
                subsets.append(subset_indices)
            else:
                subsets[i] = torch.cat([subsets[i], subset_indices])

    subsets = [Subset(dst_train, subset) for subset in subsets]

    print('subset数: ', len(subsets))
    for i, sub in enumerate(subsets):
        print('subset %d: %d' % (i, len(sub)))
        # クラス分布の確認
        targets = [dst_train[i][1] for i in sub.indices]
        print(f"Subset {i} Class Distribution: ", {t: targets.count(t) for t in set(targets)})

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    total_mean = {}
    best_5 = []
    accuracy_logging = {"mean":[], "std":[], "max_mean":[]}
    exp_per_final_acc = []

##### 実験を複数回実行する !!!
    print('==================== Start Experiments ====================\n')
    for exp in range(args.num_exp):
        total_mean[exp] = {'mean':[], 'std':[]}
        best_5.append(0)
        print('\n================== Exp %d ==================\n ' % exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        # 各サブセットでデータ蒸留を実行し、結果を保存
        all_image_syn = []
        all_label_syn = []

##### subsetごとにデータ蒸留を行う
        print('================== subsetごとにデータ蒸留を行う ==================\n ')
        for sub_idx, subset in enumerate(subsets):
            print(f'\n========== Processing Subset {sub_idx + 1}/{num_splits} ==========\n')
### これは何をしているのか？?????????????????????????
            split_loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_real, shuffle=True)

            ''' organize the real dataset '''
            images_all = []
            labels_all = []
            indices_class = [[] for c in range(num_classes)]

##### Training Dataをすべて取得 ???
            print("各サブセットのデータを取得")
            images_all = [torch.unsqueeze(subset[i][0], dim=0) for i in range(len(subset))]
            print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            print("Images All: ", len(images_all)) # MNIST:50000

            labels_all = [subset[i][1] for i in range(len(subset))]
            print("Labels All: ", len(labels_all)) # MNIST:50000

            for i, lab in enumerate(labels_all):
                indices_class[lab].append(i)
            images_all = torch.cat(images_all, dim=0).to(args.device)
            labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)


            for c in range(num_classes):
                print('class c = %d: %d real images'%(c, len(indices_class[c])))

    ##### 特定のクラスcからn個の画像をランダムに取得する関数 (後にDsynをRealで初期化する時とかで使われる)
            def get_images(c, n): # get random n images from class c
                idx_shuffle = np.random.permutation(indices_class[c])[:n]
                return images_all[idx_shuffle]

            for ch in range(channel):
                print('real images channel %d, mean = %.4f, std = %.4f' % (ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    ##### 合成データの初期化 (とりまランダムノイズで初期化し、args.initに従って初期化方法を変更する)
            ''' initialize the synthetic data '''
            image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
            label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
    ##### 合成データの数を確認
            print("合成データの数: ", len(image_syn))
            if args.init == 'real':
                print('initialize synthetic data from random real images')
                for c in range(num_classes):
                    image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
            else:
                print('initialize synthetic data from random noise')


            ''' training '''
            optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)  # optimizer_img for synthetic data
            optimizer_img.zero_grad()
            criterion = nn.CrossEntropyLoss().to(args.device)
            print('%s training begins' % get_time())

    ##### args.Iteration回のトレーニングを行う !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            max_mean = 0
            for it in range(args.Iteration + 1):

    ##### Evaluate Synthetic Data ???????
                ''' Evaluate synthetic data '''
                if it in eval_it_pool:
                    for model_eval in model_eval_pool:
                        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))
                        if args.dsa:
                            args.epoch_eval_train = 1000
                            args.dc_aug_param = None
                            print('DSA augmentation strategy: \n', args.dsa_strategy)
                            print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                        else:
                            args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)  # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                            print('DC augmentation parameters: \n', args.dc_aug_param)

                        if args.dsa or args.dc_aug_param['strategy'] != 'none':
                            args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                        else:
                            args.epoch_eval_train = 300

                        accs = []
                        for it_eval in range(args.num_eval):
                            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                            accs.append(acc_test)
                        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))

                        if it == args.Iteration:  # record the final results
                            accs_all_exps[model_eval] += accs

                    ''' visualize and save '''
                    save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                    for ch in range(channel):
                        image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                    image_syn_vis[image_syn_vis<0] = 0.0
                    image_syn_vis[image_syn_vis>1] = 1.0
                    save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.


    ##### Train Synthetic Data !!!!!!!!!!!!!!!!!!!!!
                ''' Train synthetic data '''
                net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
                net.train()
                net_parameters = list(net.parameters())
                optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
                optimizer_net.zero_grad()
                loss_avg = 0
                args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in order to be consistent with DC paper.

                for ol in range(args.outer_loop):

                    ''' freeze the running mu and sigma for BatchNorm layers '''
                    # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                    # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                    # This would make the training with BatchNorm layers easier.

                    BN_flag = False
                    BNSizePC = 16  # for batch normalization
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  # BatchNorm
                            BN_flag = True
                    if BN_flag:
                        img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                        net.train()  # for updating the mu, sigma of BatchNorm
                        output_real = net(img_real)  # get running mu, sigma
                        for module in net.modules():
                            if 'BatchNorm' in module._get_name():  # BatchNorm
                                module.eval()  # fix mu and sigma of every BatchNorm layer

                    ''' update synthetic data '''
                    loss = torch.tensor(0.0).to(args.device)
                    for c in range(num_classes):
                        img_real = get_images(c, args.batch_real)
                        lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                        img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                        lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                        if args.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                        output_real = net(img_real)
                        loss_real = criterion(output_real, lab_real)
                        gw_real = torch.autograd.grad(loss_real, net_parameters)
                        gw_real = list((_.detach().clone() for _ in gw_real))

                        output_syn = net(img_syn)
                        loss_syn = criterion(output_syn, lab_syn)
                        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                        loss += match_loss(gw_syn, gw_real, args)

                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()
                    loss_avg += loss.item()

                    if ol == args.outer_loop - 1:
                        break

                    ''' update network '''
                    image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                    dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                    trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                    for il in range(args.inner_loop):
                        epoch('train', trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)

                loss_avg /= (num_classes * args.outer_loop)

                if it % 10 == 0:
                    print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

                if it == args.Iteration:  # only record the final results
                    data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                    torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'split%d_res_%s_%s_%s_%dipc.pt' % (sub_idx + 1, args.method, args.dataset, args.model, args.ipc)))

            ##### サブセットごとに蒸留されたデータを保存
            print("append image_syn and label_syn!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            all_image_syn.append(image_syn.detach().cpu())
            all_label_syn.append(label_syn.detach().cpu())

##### Subset loop 終了
        ##### すべての蒸留データを結合
        all_image_syn = torch.cat(all_image_syn, dim=0)
        all_label_syn = torch.cat(all_label_syn, dim=0)
        print("All Image Syn: ", all_image_syn.size())
        print("All Label Syn: ", all_label_syn.size())

        ##### 評価のためのデータローダを作成
        synth_dataset = TensorDataset(all_image_syn, all_label_syn)
        synth_loader = torch.utils.data.DataLoader(synth_dataset, batch_size=args.batch_train, shuffle=False)

        ##### 評価モデルをロードし、最終的な評価を実施
        final_accs = []
        for model_eval in model_eval_pool:
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
            _, acc_train, acc_test = evaluate_synset(0, net_eval, all_image_syn, all_label_syn, testloader, args)
            final_accs.append(acc_test)

        ##### 評価結果の表示
        print("\nExp %d ,Final Evaluation on Combined Synthetic Data:"%exp)
        # print("Model_Eval_pool is : ", model_eval_pool)
        for model_eval, acc in zip(model_eval_pool, final_accs):
            print(f'Model {model_eval}: Test Accuracy = {acc:.4f}')
        exp_per_final_acc.append(final_accs)
##### 実験終了


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
    
    print('\n==================== Maximum Results ====================\n')
    
    best_means = []
    best_std = []
    for exp in total_mean.keys():
        best_idx = np.argmax(total_mean[exp]['mean'])
        best_means.append(total_mean[exp]['mean'][best_idx])
        best_std.append(total_mean[exp]['std'][best_idx])
    
    mean = np.mean(best_means)
    std = np.mean(best_std)
        
    num_eval = args.num_exp*args.num_eval
    print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model,num_eval, key, mean*100, std*100))
    
    
    print('\n==================== Top 5 Results ====================\n')
    
       
    mean = np.mean(best_5)
    std = np.std(best_5)
        
    num_eval = args.num_exp*args.num_eval
    print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model,num_eval, key, mean*100, std*100))

##### データ分割してFLにした場合は、以下の結果のみ見ればおｋ　複数実験を行った場合は、それぞれの実験の結果がリストで保存されている. meanとってもいいかもしれない
    print('\n==================== 実験ごとのAll Dsyn Test Acc ====================\n')
    print(exp_per_final_acc)
    print("一応Mean: ", np.mean(exp_per_final_acc))
if __name__ == '__main__':
    main()
