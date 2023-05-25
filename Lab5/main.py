import argparse
import random
from torch.utils.data import DataLoader
from torch import autograd
from dataset import *
from models import *
from draw import *

def gradient_penalty(D, real_samples, fake_samples, batchsz, args):

    # 对判别器的参数进行梯度惩罚
    epsilon = torch.rand(batchsz, 1).to(args.device)
    epsilon = epsilon.expand_as(real_samples)
    interpolated_samples = real_samples + epsilon * (fake_samples - real_samples)
    interpolated_samples.requires_grad_()
    interpolated_outputs = D(interpolated_samples)
    grad_outputs = torch.ones_like(interpolated_outputs).to(args.device)
    gradients = autograd.grad(outputs=interpolated_outputs, inputs=interpolated_samples,
                              grad_outputs=grad_outputs, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradient_penalty = torch.pow(gradients.norm(2, dim=1) - 1, 2).mean()

    return gradient_penalty

def main(args):

    global loss_D, loss_G
    device = args.device

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("\nProcessing " + args.dataset + " dataset...")
    if args.dataset == "points":
        dataset = Points()
        train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    else:
        raise ValueError(f"dataset {args.dataset} not supported")
    print("Data processing finished!")

    print("\nBuilding model " + args.model + "...")
    if args.model in ["GAN", "WGAN", "WGAN-GP"]:
        G = Generator()
        D = Discriminator()
    else:
        raise ValueError(f"dataset {args.model} not supported")
    G.to(device)
    D.to(device)
    print("Model building finished!")

    if args.optimizer == 'adam':
        optim_G = torch.optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
        optim_D = torch.optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))
    elif args.optimizer == 'sgd':
        optim_G = torch.optim.Adam(G.parameters(), lr=3e-4)
        optim_D = torch.optim.Adam(D.parameters(), lr=3e-4)
    elif args.optimizer == 'rmsprop':
        optim_G = torch.optim.RMSprop(G.parameters(), lr=1e-4)
        optim_D = torch.optim.RMSprop(D.parameters(), lr=1e-4)
    else:
        raise ValueError(f"dataset {args.optimizer} not supported")

    print("\nStart training...")
    all_loss = []

    # 训练过程
    for epoch in range(1, args.epochs+1):
        for idx, real_samples in enumerate(train_loader):
            # 生成真实样本和生成样本
            real_samples = real_samples.to(device)
            noise = torch.randn(real_samples.shape[0], 10).to(device)
            fake_samples = G(noise).detach()

            # 计算真实样本和生成样本的判别器输出
            real_outputs = D(real_samples)
            fake_outputs = D(fake_samples)

            # 计算判别器损失
            if args.model == 'GAN':
                D_loss = - (torch.log(real_outputs) + torch.log(1. - fake_outputs)).mean()
            else:
                D_loss = torch.mean(fake_outputs) - torch.mean(real_outputs)

            # 清零判别器的梯度
            optim_D.zero_grad()
            # 反向传播和优化判别器
            D_loss.backward()
            optim_D.step()

            # 限制判别器的权重范围
            if args.model == 'WGAN':
                for p in D.parameters():
                    p.data.clamp_(-args.CLAMP, args.CLAMP)

            if args.model == 'WGAN-GP':
                D_loss += 0.2 * gradient_penalty(D, real_samples, fake_samples.detach(), real_samples.shape[0], args)

            if idx % 2 == 0:
                # 生成新的生成样本
                z = torch.randn(args.batch_size, 10).to(device)
                fake_samples = G(z)
                # 计算生成样本的判别器输出
                fake_outputs = D(fake_samples)
                # 计算生成器损失
                if args.model == 'GAN':
                    G_loss = torch.log(1. - fake_outputs).mean()
                else:
                    G_loss = -torch.mean(fake_outputs)
                # 清零生成器的梯度
                optim_G.zero_grad()
                # 反向传播和优化生成器
                G_loss.backward()
                optim_G.step()

        if epoch % 5 == 0:
            print('[epoch %d/%d] Discriminator loss: %.3f, Generator loss: %.3f'
                  % (epoch, args.epochs, D_loss.item(), G_loss.item()))
            all_loss.append([D_loss.item(), G_loss.item()])
        if epoch % 50 == 0 and args.draw:
            input = torch.randn(1000, 10).to(device)
            output = G(input)
            output = output.to('cpu').detach()
            xy = np.array(output)
            draw_scatter(D, xy, epoch, args.model)

    # draw the loss
    all_loss = np.array(all_loss)
    x = np.arange(len(all_loss))
    y1 = all_loss[:, 0]
    y2 = all_loss[:, 1]
    fig = plt.figure(2, figsize=(16, 16), dpi=150)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(x, y1, 'r', label='loss_D')
    ax2.plot(x, y2, 'g', label='loss_G')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    plt.savefig(args.output_path + args.model + "/loss.jpg")
    # save the model
    state = {"model_D": D.state_dict(), "model_G": G.state_dict()}
    torch.save(state, args.output_path + 'models/' + args.model + '.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("lab5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="WGAN-GP")  # GAN WGAN WGAN-GP
    parser.add_argument("--epochs", default=1000)
    parser.add_argument("--batch-size", default=2000)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--dataset", default="points")
    parser.add_argument("--output-path", default="./result/")
    parser.add_argument("--hidden-size", default=128)
    parser.add_argument("--input-size", default=128)
    parser.add_argument("--CLAMP", default=0.1)
    parser.add_argument("--optimizer", default="rmsprop")  # adam sgd rmsprop
    parser.add_argument("--draw", default=True, help="draw the loss and process")

    args = parser.parse_args()
    print(args)

    main(args)


