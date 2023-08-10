from models.models import Generator,Discriminator
from torch.utils.data import DataLoader,TensorDataset
from data_process import get_one_hot
import torch.autograd as autograd
import utils
import torch
import time
import os, math, argparse

class WGAN_GP():
    def __init__(self, batch_size=16, lr=0.0001, num_epochs=2000, seq_len = 200, data_dir='../train_datas/gan/enhancer.txt', \
        model_name='gan', hidden=256, device_id=0):
        self.model_name = model_name
        self.hidden = hidden
        self.batch_size = int(batch_size)
        self.lr = lr
        self.epoch = num_epochs
        self.seq_len = seq_len
        self.lambda_gp = 10
        self.noise_size = 128
        self.device = utils.get_all_gpus()[int(device_id)]
        self.checkpoints_path = '../saved_models/' + model_name + "/"
        self.output_dir = '../output/' + model_name + "/"
        self.dataset = self.get_dataset(data_dir)
        if not os.path.exists(self.checkpoints_path): os.makedirs(self.checkpoints_path)
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        self.init_model()

    def init_model(self):
        self.G = Generator(self.seq_len, self.hidden, self.noise_size).to(self.device)
        self.D = Discriminator(self.seq_len, self.hidden).to(self.device)
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))

    def get_dataset(self, data_dir):
        print("start load dataset...")
        file = open(data_dir)

        out = []
        samples_size = 0
        for line in file.readlines():
            line = line.split()[0]
            if "N" not in line:
                out.append(get_one_hot(line, self.seq_len))
                samples_size+=1

        self.sample_size = samples_size
        x = torch.stack(out, dim=0)
        y = torch.ones(x.shape[0])

        # 把张量放入dataloater中
        dataset = TensorDataset(x, y)
        print("loading completed！")
        file.close()

        return dataset

    def compute_gradient_penalty(self, real_samples, fake_samples, device):
        """Calculates the gradient penalty loss for wgan GP"""
        alpha = torch.rand(self.batch_size, 1, 1, device=device)
        alpha = alpha.view(-1, 1, 1)
        alpha = alpha.expand_as(real_samples)
        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)

        # interpolates = interpolates.cuda() if self.use_cuda else interpolates
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                                  create_graph=True,
                                  retain_graph=True
                                  )[0]

        gradients = gradients.contiguous().view(self.batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # gradient_penalty = ((gradients.norm(2, dim=1).norm(2,dim=1) - 1) ** 2).mean() * self.lamda
        return ((gradients_norm - 1) ** 2).mean()

    def save_models(self, i, D_loss, G_loss, fake_loss, real_loss, w, gp):
        torch.save({
            'epoch': i + 1,
            'modelG_state_dict': self.G.state_dict(),
            'modelD_state_dict': self.D.state_dict(),
            'optimizerG_state_dict': self.G_optimizer.state_dict(),
            'optimizerD_state_dict': self.D_optimizer.state_dict(),
            'D_loss':D_loss,
            'G_loss':G_loss,
            'fake_loss':fake_loss,
            'real_loss':real_loss,
            'w':w,
            'gp':gp
        }, (self.checkpoints_path + self.model_name + "_epoch" + str(i+1) + ".pth"))

    def load_checkpoints(self):
        with open(self.checkpoints_path + 'checkpoint_epoch_num.txt', 'r') as file:
            epoch = file.read().replace('\n', '')

        checkpoint = torch.load(self.checkpoints_path + self.model_name + "_epoch" + epoch + ".pth")
        self.G.load_state_dict(checkpoint['modelG_state_dict'])
        self.G_optimizer.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.D.load_state_dict(checkpoint['modelD_state_dict'])
        self.D_optimizer.load_state_dict(checkpoint['optimizerD_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        print("checkpoint_epoch: ", checkpoint_epoch)
        return checkpoint_epoch, checkpoint

    def train(self):
        # 加载检查点
        checkpoint_epoch = 0
        D_loss_for_draw, G_loss_for_draw, Fake_loss, Real_loss, W, GP = [], [], [], [], [], []
        if os.path.exists(self.checkpoints_path+"checkpoint_epoch_num.txt"):
            checkpoint_epoch, checkpoint = self.load_checkpoints()
            D_loss_for_draw, G_loss_for_draw, Fake_loss, Real_loss, W, GP = checkpoint['D_loss'], checkpoint['G_loss'], checkpoint['fake_loss'], checkpoint['real_loss'], checkpoint['w'], checkpoint['gp']

        counter = 0
        total_batch_num = int(self.sample_size / self.batch_size)
        train_dataset = self.dataset
        dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=2, drop_last=True)
        for i in range(checkpoint_epoch, self.epoch):
            self.G.train()
            self.D.train()
            D_epoch_loss, G_epoch_loss, grad_penality, fake_loss, real_loss, w = 0, 0, 0, 0, 0, 0
            for batch_num, (x, y) in enumerate(dataloader):
                x = x.to(self.device)

                self.D_optimizer.zero_grad()
                # torch.manual_seed(0)
                random_noise = torch.randn(self.batch_size, self.noise_size, device=self.device)
                fake_data = self.G(random_noise)
                d_fake_pred = self.D(fake_data)
                fake_score = d_fake_pred.mean()
                d_real_pred = self.D(x)
                real_score = d_real_pred.mean()

                gp = self.compute_gradient_penalty(x, fake_data, device=self.device) * self.lambda_gp
                D_loss = fake_score - real_score + gp
                D_loss.backward()
                self.D_optimizer.step()
                Wasserstein_D = real_score - fake_score

                if counter % 10 == 0:
                    self.G.zero_grad()
                    random_noise = torch.randn(self.batch_size, self.noise_size, device=self.device)
                    fake_seq = self.G(random_noise)
                    output = self.D(fake_seq)
                    G_loss = -torch.mean(output)
                    G_loss.backward()
                    self.G_optimizer.step()

                counter += 1
                with torch.no_grad():
                    print('d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f} , D(G(z)): {:.2f} , W: {:.4f}, GP: {:.2f}'
                          .format(D_loss.item(), G_loss.item(),
                                  real_score.item(), fake_score.item(),
                                  Wasserstein_D.item(), gp.item()))

                    D_epoch_loss += D_loss.item()
                    G_epoch_loss += G_loss.item()
                    real_loss += real_score.item()
                    fake_loss += fake_score.item()
                    grad_penality += gp.item()
                    w += math.fabs(Wasserstein_D.item())

            with torch.no_grad():
                D_epoch_loss /= total_batch_num
                G_epoch_loss /= total_batch_num
                real_loss /= total_batch_num
                fake_loss /= total_batch_num
                grad_penality /= total_batch_num
                w /= total_batch_num

            D_loss_for_draw.append(D_epoch_loss)
            G_loss_for_draw.append(G_epoch_loss)
            Real_loss.append(real_loss)
            Fake_loss.append(fake_loss)
            GP.append(grad_penality)
            W.append(w)
            print("epoch:", i, ", D_loss:", D_epoch_loss, ", G_loss:", G_epoch_loss)

            if (i != 0 and (i+1) % 100 == 0) or (i!=0 and i == (self.epoch - 1)):
                with open(self.checkpoints_path + 'checkpoint_epoch_num.txt', 'w') as file:
                    file.write(str(i+1))
                self.save_models(i, D_loss_for_draw, G_loss_for_draw, Fake_loss, Real_loss, W, GP)

        print("end!")
        # 为结果绘制图片
        title = "BatchSize:" + str(self.batch_size) + ", D_lr:" + str(self.D_optimizer.defaults['lr'])
        utils.draw(D_loss_for_draw, 'Discriminator', self.output_dir, title)
        title = "BatchSize:" + str(self.batch_size) + ", G_lr:" + str(self.G_optimizer.defaults['lr'])
        utils.draw(G_loss_for_draw, 'Generator', self.output_dir, title)
        utils.draw(W, 'Wasserstein distance', self.output_dir)
        utils.draw(GP, 'gradient penality', self.output_dir)
        utils.draw1(Real_loss, Fake_loss, self.output_dir,title=None, legend=['D real loss', 'D fake loss'])

def main(argv=None):
    parser = argparse.ArgumentParser(description='WGAN_GP for producing enhancer sequences.')
    parser.add_argument("--model_name", default="wgan-gpr", help="model name for checkpoint and output")
    parser.add_argument("--gpu_id", default="0", help="Which GPU to use for training(if you have gpu)")
    parser.add_argument("--data_dir", default='../train_datas/gan/enhancer.txt', help="training data path")
    parser.add_argument("--batch_size", default=16 , help="batch size")
    args = parser.parse_args()
    start = time.time()
    print("init models...")
    wgan_gp = WGAN_GP(model_name=args.model_name, device_id=args.gpu_id, data_dir=args.data_dir, batch_size=args.batch_size)
    print("init complete!")
    wgan_gp.train()
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    main()