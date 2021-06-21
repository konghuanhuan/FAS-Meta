import itertools
import os
from collections import OrderedDict
import torch
import torch.optim as optim
from torch import nn
from torch.nn import DataParallel
import random

def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for catimages, labels in data_loader:
            yield (catimages, labels)


def Train(args, HeadModel, TailModel,
          data_loader1_real, data_loader1_fake,
          data_loader2_real, data_loader2_fake,
          data_loader3_real, data_loader3_fake,
          savefilename):

    HeadModel.train()
    TailModel.train()

    HeadModel = DataParallel(HeadModel)
    criterionCls = nn.BCEWithLogitsLoss()

    optimizer_all = optim.Adam(itertools.chain(HeadModel.parameters(), TailModel.parameters()),
                               lr=args.lr_meta, betas=(args.beta1, args.beta2))

    iternum = max(len(data_loader1_real), len(data_loader1_fake),
                  len(data_loader2_real), len(data_loader2_fake),
                  len(data_loader3_real), len(data_loader3_fake))

    print('iternum={}'.format(iternum))

    # train network #
    global_step = 0
    ModelHeadName = ''
    ModelTailName = ''
    for epoch in range(args.epochs):

        data1_real = get_inf_iterator(data_loader1_real)
        data1_fake = get_inf_iterator(data_loader1_fake)
        data2_real = get_inf_iterator(data_loader2_real)
        data2_fake = get_inf_iterator(data_loader2_fake)
        data3_real = get_inf_iterator(data_loader3_real)
        data3_fake = get_inf_iterator(data_loader3_fake)

        for step in range(iternum):
            # ============ one batch extraction ============#
            cat_img1_real, lab1_real = next(data1_real)
            cat_img1_fake, lab1_fake = next(data1_fake)
            cat_img2_real, lab2_real = next(data2_real)
            cat_img2_fake, lab2_fake = next(data2_fake)
            cat_img3_real, lab3_real = next(data3_real)
            cat_img3_fake, lab3_fake = next(data3_fake)

            # ============ one batch collection ============#

            catimg1 = torch.cat([cat_img1_real, cat_img1_fake], 0).cuda()
            lab1 = torch.cat([lab1_real, lab1_fake], 0).float().cuda()
            catimg2 = torch.cat([cat_img2_real, cat_img2_fake], 0).cuda()
            lab2 = torch.cat([lab2_real, lab2_fake], 0).float().cuda()
            catimg3 = torch.cat([cat_img3_real, cat_img3_fake], 0).cuda()
            lab3 = torch.cat([lab3_real, lab3_fake], 0).float().cuda()

            # ============ doamin list augmentation ============#
            catimglist = [catimg1, catimg2, catimg3]
            lablist = [lab1, lab2, lab3]

            domain_list = list(range(len(catimglist)))
            random.shuffle(domain_list)

            meta_train_list = domain_list[:args.metatrainsize]
            meta_test_list = domain_list[args.metatrainsize:]

            # ============ meta training ============#

            Loss_cls_train = 0.0
            adapted_state_dicts = []
            for index in meta_train_list:

                catimg_meta = catimglist[index]
                lab_meta = lablist[index]
                batchidx = list(range(len(catimg_meta)))
                random.shuffle(batchidx)
                img_rand = catimg_meta[batchidx, :]
                lab_rand = lab_meta[batchidx]
                feat_ext_all, feat = HeadModel(img_rand)
                pred = TailModel(feat)
                Loss_cls = criterionCls(pred.squeeze(), lab_rand)
                Loss_cls_train += Loss_cls

                zero_param_grad(TailModel.parameters())
                grads_ModelTail = torch.autograd.grad(Loss_cls, TailModel.parameters(), create_graph=True)
                fast_weights_ModelTail = TailModel.cloned_state_dict()

                adapted_params = OrderedDict()
                for (key, val), grad in zip(TailModel.named_parameters(), grads_ModelTail):
                    adapted_params[key] = val - args.meta_step_size * grad
                    fast_weights_ModelTail[key] = adapted_params[key]

                adapted_state_dicts.append(fast_weights_ModelTail)

            # ============ meta testing ============#
            Loss_cls_test = 0.0

            index = meta_test_list[0]
            catimg_meta = catimglist[index]
            lab_meta = lablist[index]
            batchidx = list(range(len(catimg_meta)))
            random.shuffle(batchidx)

            img_rand = catimg_meta[batchidx, :]
            lab_rand = lab_meta[batchidx]

            feat_ext_all, feat = HeadModel(img_rand)

            for n_scr in range(len(meta_train_list)):
                a_dict = adapted_state_dicts[n_scr]
                pred = TailModel(feat, a_dict)
                Loss_cls = criterionCls(pred.squeeze(), lab_rand)
                Loss_cls_test += Loss_cls

            Loss_meta_train = Loss_cls_train
            Loss_meta_test = Loss_cls_test

            Loss_all = Loss_meta_train + args.W_metatest * Loss_meta_test

            optimizer_all.zero_grad()
            Loss_all.backward()
            optimizer_all.step()

            global_step += 1

            if (step + 1) % args.show_step == 0:
                print('Loss_meta_train: %.8f'%(Loss_meta_train.item()))
                print('Loss_meta_test: %.8f' % (Loss_meta_test.item()))
                print('Loss_cls_train: %.8f' % (Loss_cls_train.item()))
                print('Loss_cls_test: %.8f' % (Loss_cls_test.item()))

        # save model parameters #
        if (epoch + 1) % args.model_save_epoch == 0:
            model_save_path = os.path.join(args.results_path, 'snapshots', savefilename)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            ModelHeadName = os.path.join(model_save_path, 'ModelHead-{}.pt'.format(epoch + 1))
            ModelTailName = os.path.join(model_save_path, 'ModelTail-{}.pt'.format(epoch + 1))

            torch.save(HeadModel.state_dict(), ModelHeadName)
            torch.save(TailModel.state_dict(), ModelTailName)

    return ModelHeadName, ModelTailName



def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
