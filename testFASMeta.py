from DatasetsBox import test_dataset_loader
import argparse
import torch
import FASNet
import os


def model_test_csv(dataset_target, ModelHead_restore,ModelTail_restore,savename,imgroot):
    batchsize = 10
    data_loader_target = test_dataset_loader(imgroot=imgroot, name=dataset_target, batch_size=batchsize)
    modelhead =FASNet.ModelHead()
    modeltail = FASNet.ModelTail(momentum=1)

    ModelHead = FASNet.init_model(net=modelhead, restore=ModelHead_restore, parallel_reload=True)
    ModelTail = FASNet.init_model(net=modeltail, restore=ModelTail_restore, parallel_reload=False)

    ModelHead.eval()
    ModelTail.eval()
    score_list = []
    name_list = []
    for (catimages, names) in data_loader_target:
        images = catimages.cuda()
        _, feat = ModelHead(images)
        label_pred = ModelTail(feat)
        score = torch.sigmoid(label_pred).cpu().detach().numpy()
        score_list += score.squeeze().tolist()
        name_list += names

    with open(savename,'w') as fp:
        for ind,name in enumerate(name_list):
            prename = os.path.basename(name)
            fp.write('%s %.4f\n' % (prename, score_list[ind]))




if __name__ == '__main__':
    print('start')
    parser = argparse.ArgumentParser(description="FAS_Meta_test")
    parser.add_argument('--imgroot', type=str,
                        default='/home/storage/storage48/data2/konglingmei/program/3DMaskDetect/datasets')
    parser.add_argument('--modelpath', type=str, default='/home/storage/storage48/data2/konglingmei/program/3DMaskDetect/models/FASMeta/Train_20210621')
    parser.add_argument('--savepath', type=str,default='./results')
    args = parser.parse_args()

    savepath = args.savepath
    imgroot = args.imgroot
    modelpath = os.path.join(args.modelpath, 'snapshots/1463')
    prefix = '-80.pt'
    ModelHead_restore = os.path.join(modelpath,'ModelHead-{}.pt'.format(prefix))
    ModelTail_restore = os.path.join(modelpath,'ModelTail-{}.pt'.format(prefix))
    # savepath = './results'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    dataset_target = 'val'
    valname = os.path.join(savepath,'val'+prefix+'.txt')
    model_test_csv(dataset_target, ModelHead_restore, ModelTail_restore, valname, imgroot)

    dataset_target = 'test'
    testname = os.path.join(savepath, 'test' + prefix + '.txt')
    model_test_csv(dataset_target, ModelHead_restore, ModelTail_restore, testname, imgroot)

    names = []
    pres = []

    #val
    with open(valname,'r') as fp:
        contents = fp.readlines()

    labelDicts = {}
    for content in contents:
        line = content.split()
        labelDicts[line[0]] = float(line[1])

    for ind in range(1,4646):
        name = '%04d.png'%(ind)
        if name in labelDicts:
            pre = labelDicts[name]
        else:
            pre = 0.0
        names.append(name)
        pres.append(pre)

    #test
    with open(testname,'r') as fp:
        contents = fp.readlines()

    labelDicts = {}
    for content in contents:
        line = content.split()
        labelDicts[line[0]] = float(line[1])

    for ind in range(1,173621):
        name = '%04d.png'%(ind)
        if name in labelDicts:
            pre = labelDicts[name]
        else:
            pre = 0.0
        names.append(name)
        pres.append(pre)

    #merge
    valtestname = os.path.join(savepath,'valtest.txt')
    with open(valtestname, 'w') as fp:
        for ind, name in enumerate(names):
            fp.write('%s %.4f\n'%(name, pres[ind]))



