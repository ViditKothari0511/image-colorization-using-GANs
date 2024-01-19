import cgan
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from PIL import Image



class CreateDataset(Dataset):
    transform_image = transforms.Compose([transforms.ToTensor()])

    def __init__(self, colored_path, sketch_path):
        self.colored_path = colored_path
        self.sketch_path = sketch_path

    def __len__(self):
        return 2

    def __getitem__(self, item):
        true_image = Image.open(self.colored_path)
        true_image = self.transform_image(true_image.resize((512, 512)))
        gen_image = Image.open(self.sketch_path)
        gen_image = self.transform_image(gen_image.resize((512, 512)))
        return {'true_image': true_image[:3, :, :], 'gen_image': gen_image[:3, :, :]}


def train(
        gen_model: nn.Module,
        dis_model: nn.Module,
        train_loader: DataLoader,
        checkpoint_file: Path,
        num_epochs: int = 100,
        device: torch.device = torch.device("cpu"),
        lr: float = 0.01,
):
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'DEVICE = {device}')
    gen_model.to(device)
    dis_model.to(device)
    d_optimizer = optim.Adam(dis_model.parameters(), betas=(0.5, 0.999), lr=lr)
    g_optimizer = optim.Adam(gen_model.parameters(), betas=(0.5, 0.999), lr=0.005)
    d_criterion = nn.BCELoss()
    g_criterion_1 = nn.BCELoss()
    g_criterion_2 = nn.L1Loss()
    smoothing = 0.2
    for ep in range(num_epochs):
        for i, data in enumerate(train_loader):
            images = data
            true_image = images['true_image'].to(device)
            gen_image = images['gen_image'].to(device)
            fake_image = gen_model(gen_image)
            fake_image = F.interpolate(fake_image, size=(512, 512), mode='bilinear', align_corners=False)
            d_optimizer.zero_grad()
            target_size = (512, 512)
            real_output = dis_model(true_image, gen_image)
            real_label = (1-smoothing)*torch.ones_like(real_output).to(device)
            real_loss = d_criterion(real_output, real_label)
            fake_output = dis_model(fake_image.detach(), gen_image)
            fake_label = torch.zeros_like(fake_output).to(device)
            fake_loss = d_criterion(fake_output, fake_label)
            d_loss = real_loss + fake_loss
            d_loss.backward(retain_graph=True)
            d_optimizer.step()
            g_optimizer.zero_grad()
            fake_output = dis_model(fake_image, gen_image)
            real_label = torch.ones_like(fake_output).to(device)
            g_loss_1 = g_criterion_1(fake_output, real_label)
            g_loss_2 = g_criterion_2(fake_image, true_image)
            g_loss = g_loss_1 - g_loss_2
            g_loss.backward()
            g_optimizer.step()
        print(f'Epoch: {ep} D Loss: {d_loss} G Loss: {g_loss}')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': gen_model.state_dict(),
        'optimizer_state_dict': g_optimizer.state_dict(),
        'loss': g_loss,
        'name': 'gen_model'
    }, 'gen_model.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': dis_model.state_dict(),
        'optimizer_state_dict': d_optimizer.state_dict(),
        'loss': d_loss,
        'name': 'dis_model'
    }, 'dis_model.pt')


def test(image_path, model, device):
    image = Image.open(image_path)
    image_tensor = transforms.ToTensor()(image.resize((512, 512)))
    model.load_state_dict(torch.load('gen_model.pt')['model_state_dict'])
    model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)
    output = model(image_tensor[:3, :, :].unsqueeze(0))
    return output


def get_manga_images(**kwargs):
    train_set = CreateDataset(colored_path='train/color.jpg', sketch_path='test/sketch.jpg')

    train_loader = DataLoader(train_set, shuffle=False, **kwargs)

    return train_loader



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", type=Path, default=None)
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train_loader = get_manga_images()
    gen_model = cgan.Generator()
    dis_model = cgan.Discriminator()
    if args.checkpoint_file is None:
        args.checkpoint_file = Path("logs") / "checkpoint.pt"

    args.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    train(
        gen_model=gen_model,
        dis_model=dis_model,
        train_loader=train_loader,
        checkpoint_file=args.checkpoint_file,
        num_epochs=args.num_epochs,
        device=torch.device(args.device),
        lr=args.lr
    )

    image_path = 'test/sketch.jpg'
    model = cgan.Generator()
    device = torch.device('cpu')
    output = test(image_path, model, device)
    output = output.detach().numpy()
    output = output[0].transpose(1, 2, 0)
    output = output * 255
    output = output.astype('uint8')
    output = Image.fromarray(output)
    output.save('generated_image.jpg')
