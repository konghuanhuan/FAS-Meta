import os.path as osp
import argparse

from trainEpoch import Train
from DatasetsBox import train_dataset_loader
import FASNet
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FAS_Meta")
    # datasets
    parser.add_argument('--imgroot', type=str, default='/home/storage/storage48/data2/konglingmei/program/3DMaskDetect/datasets')
    parser.add_argument('--dataset1', type=str, default='1')
    parser.add_argument('--dataset2', type=str, default='4')
    parser.add_argument('--dataset3', type=str, default='6')


    parser.add_argument('--metatrainsize', type=int, default=2)
    # optimizer
    parser.add_argument('--meta_step_size', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--bn_momentum', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # training configs
    parser.add_argument('--results_path', type=str,
                        default='/home/storage/storage48/data2/konglingmei/program/3DMaskDetect/models/FASMeta/Train_20210621')
    parser.add_argument('--batchsize', type=int, default=10)

    parser.add_argument('--show_step', type=int, default=500)
    parser.add_argument('--model_save_epoch', type=int, default=5)

    parser.add_argument('--W_metatest', type=int, default=1)
    args = parser.parse_args()

    levels = [0, 1, 2]
    # epochs = [80, 60, 50]
    epochs = [80, 80, 80]

    ModelHeadName = ''
    ModelTailName = ''
    args.lr_meta = 1e-3

    for index,level in enumerate(levels):
        args.epochs = epochs[index]
        # args.lr_meta *= 0.5
        savefilename = osp.join(args.dataset1 + args.dataset2 + args.dataset3 + str(level))
        data_loader1_real = train_dataset_loader(imgroot=args.imgroot, name=args.dataset1, getreal=True,
                                                 batch_size=args.batchsize, trainlevel=level)
        data_loader1_fake = train_dataset_loader(imgroot=args.imgroot, name=args.dataset1, getreal=False,
                                                 batch_size=args.batchsize, trainlevel=level)
        data_loader2_real = train_dataset_loader(imgroot=args.imgroot, name=args.dataset2, getreal=True,
                                                 batch_size=args.batchsize, trainlevel=level)
        data_loader2_fake = train_dataset_loader(imgroot=args.imgroot, name=args.dataset2, getreal=False,
                                                 batch_size=args.batchsize, trainlevel=level)
        data_loader3_real = train_dataset_loader(imgroot=args.imgroot, name=args.dataset3, getreal=True,
                                                 batch_size=args.batchsize, trainlevel=level)
        data_loader3_fake = train_dataset_loader(imgroot=args.imgroot, name=args.dataset3, getreal=False,
                                                 batch_size=args.batchsize, trainlevel=level)

        modelhead = FASNet.ModelHead()
        modeltail = FASNet.ModelTail(momentum=args.bn_momentum)

        if ModelHeadName != '':
            ModelHead_restore = ModelHeadName
            ModelTail_restore = ModelTailName

        else:
            ModelHead_restore = None
            ModelTail_restore = None

        ModelHead = FASNet.init_model(net=modelhead, restore=ModelHead_restore, parallel_reload=True)
        ModelTail = FASNet.init_model(net=modeltail, restore=ModelTail_restore, parallel_reload=False)

        ModelHeadName, ModelTailName = \
            Train(args, ModelHead, ModelTail,
              data_loader1_real, data_loader1_fake,
              data_loader2_real, data_loader2_fake,
              data_loader3_real, data_loader3_fake,
              savefilename)




