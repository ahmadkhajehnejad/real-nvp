"""Train Real NVP on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util

from models import RealNVP, RealNVPLoss
from tqdm import tqdm
import numpy as np
import copy
from PIL import Image

DATASET = 'mnist' # 'cifar10  #
N_TRAIN = 300
N_VAL = 75
N_TEST = 1000
N_NOISY_SAMPLES_PER_TEST_SAMPLE = 100
PATIENCE = 20


class MyToTensor(object):
    def __call__(self, pic):
        pic = np.array(pic).astype(np.float32) / 256
        if pic.ndim < 3:
            pic = pic[:, :, None]
        return torch.from_numpy(pic.transpose((2, 0, 1)))


    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_noisy_data(data, targets):
    noisy_samples_np = np.random.rand(data.shape[0] * N_NOISY_SAMPLES_PER_TEST_SAMPLE * 28 * 28) * 1.0
    noisy_samples = torch.from_numpy(noisy_samples_np.reshape([-1,28,28]).astype(np.float32))
    noisy_data = data.repeat(N_NOISY_SAMPLES_PER_TEST_SAMPLE, 1, 1).float() + noisy_samples
    noisy_targets = targets.repeat(N_NOISY_SAMPLES_PER_TEST_SAMPLE)
    return noisy_data, noisy_targets


def main(args):
    global best_loss
    global cnt_early_stop

    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
    start_epoch = 0

    # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        MyToTensor()
        # transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        MyToTensor()
        #transforms.ToTensor()
    ])

    assert DATASET in ['mnist', 'cifar10']
    if DATASET == 'mnist':
        dataset_picker = torchvision.datasets.MNIST
    else:
        dataset_picker = torchvision.datasets.CIFAR10

    trainset = dataset_picker(root='data', train=True, download=True, transform=transform_train)
    testset = dataset_picker(root='data', train=False, download=True, transform=transform_test)

    valset = copy.deepcopy(trainset)

    train_val_idx = np.random.choice(np.arange(trainset.data.shape[0]), size=N_TRAIN+N_VAL, replace=False)
    train_idx = train_val_idx[:N_TRAIN]
    val_idx = train_val_idx[N_TRAIN:]
    valset.data = valset.data[val_idx]
    trainset.data = trainset.data[train_idx]


    test_idx = np.random.choice(np.arange(testset.data.shape[0]), size=N_TEST, replace=False)
    testset.data = testset.data[test_idx]

    if DATASET == 'mnist':
        trainset.targets = trainset.targets[train_idx]
        valset.targets = valset.targets[val_idx]
        testset.targets = testset.targets[test_idx]
    else:
        trainset.targets = np.array(trainset.targets)[train_idx]
        valset.targets = np.array(valset.targets)[val_idx]
        testset.targets = np.array(testset.targets)[test_idx]

    # noisytestset = copy.deepcopy(testset)
    if DATASET == 'mnist':
        trainset.data, trainset.targets = get_noisy_data(trainset.data, trainset.targets)
        valset.data, valset.targets = get_noisy_data(valset.data, valset.targets)
        testset.data, testset.targets = get_noisy_data(testset.data, testset.targets)

    else:
        noisy_samples = np.random.rand(N_TEST * N_NOISY_SAMPLES_PER_TEST_SAMPLE, 32, 32, 3) - 0.5
        noisytestset.data = np.tile( noisytestset.data, (N_NOISY_SAMPLES_PER_TEST_SAMPLE, 1, 1, 1)) + noisy_samples
        noisytestset.targets = np.tile(noisytestset.targets, [N_NOISY_SAMPLES_PER_TEST_SAMPLE])

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # noisytestloader = data.DataLoader(noisytestset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    print('Building model..')
    if DATASET == 'mnist':
        net = RealNVP(num_scales=2, in_channels=1, mid_channels=64, num_blocks=8)
    else:
        net = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['val_loss']
        start_epoch = checkpoint['epoch']

    loss_fn = RealNVPLoss()
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm)
        prev_best_loss = best_loss
        test(epoch, net, valloader, device, loss_fn, args.num_samples, 'val')
        if best_loss < prev_best_loss:
            cnt_early_stop = 0
        else:
            cnt_early_stop += 1
        if cnt_early_stop >= PATIENCE:
            break
        # test(epoch, net, testloader, device, loss_fn, args.num_samples, 'test')
        # test(epoch, net, noisytestloader, device, loss_fn, args.num_samples, 'noisytest')

    checkpoint = torch.load('ckpts/best.pth.tar')
    net.load_state_dict(checkpoint['net'])

    pixelwise_ll = -pixelwise_test(net, testloader, device, loss_fn, args.num_samples)
    pixelwise_ll = pixelwise_ll.reshape([-1, 28, 28])
    os.makedirs('pixelwise_loglikelihood', exist_ok=True)
    for i in range(len(pixelwise_ll)):
        tmp = np.exp( pixelwise_ll[i] )
        tmp = 255 * ( tmp / np.max(tmp) )
        im = Image.fromarray(tmp)
        im.convert('RGB').save('pixelwise_loglikelihood/' + str(i) + '.png')
    for i, (x,_) in enumerate(testloader):
        x_np = np.array(x.cpu(), dtype=np.float)
        for j in range(args.num_samples):
            im = Image.fromarray(255 * x_np[j].reshape([28,28]))
            im.convert('RGB').save('pixelwise_loglikelihood/' + str(j) + '-orig.png')
        break
    test(epoch, net, testloader, device, loss_fn, args.num_samples, 'test')
    # test(epoch, net, noisytestloader, device, loss_fn, args.num_samples, 'noisytest')


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))

        print('\ntrain loss = ', loss_meter.avg)
        print('train pbd = ', util.bits_per_dim(x, loss_meter.avg), '\n')


def sample(net, batch_size, device):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """

    if DATASET == 'mnist':
        z = torch.randn((batch_size, 1, 32, 32), dtype=torch.float32, device=device)
    else:
        z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x


def test(epoch, net, testloader, device, loss_fn, num_samples, label):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, _ in testloader:
                #if True: # label.endswith('test'):
                #    print(x.shape, x.type(), x.min(), x.max())
                x = x.to(device)
                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, sldj)
                loss_meter.update(loss.item(), x.size(0))
                progress_bar.set_postfix(loss=loss_meter.avg,
                                         bpd=util.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))
            print('\n' + label + ' loss = ', loss_meter.avg)
            print(label + ' pbd = ', util.bits_per_dim(x, loss_meter.avg), '\n')

    # Save checkpoint
    if label == 'val' and loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'val_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg

    # Save samples and data
    images = sample(net, num_samples, device)
    os.makedirs('samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))

def pixelwise_test( net, testloader, device, loss_fn, num_samples):
    net.eval()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)
                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, sldj, True)
                break

    loss = np.array(loss.cpu(), dtype=np.float)
    return loss[:num_samples]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
    parser.add_argument('--num_epochs', default=1000, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')

    best_loss = np.Inf
    cnt_early_stop = 0

    main(parser.parse_args())

