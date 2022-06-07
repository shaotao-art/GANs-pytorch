import torch
from torch import nn
from model_modules import ConvBlock, WSConv2d, UpSampling


z_dim = 256
final_img_size = 256
# in_channels_lst = [z_dim, z_dim, z_dim, z_dim, int(z_dim/2), int(z_dim/4)]
# out_channels_lst = [z_dim, z_dim, z_dim, int(z_dim/2), int(z_dim/4), int(z_dim/8)]

in_channels_lst = [256, 256, 256, 256, 128, 64]
out_channels_lst = [256, 256, 256, 128, 64, 32]

class Gen(nn.Module):
    def __init__(self, z_dim=256, img_channel=3) -> None:
        super(Gen, self).__init__()
        self.init_conv = nn.Sequential(
            nn.ConvTranspose2d(z_dim, z_dim, 4, 1, 0),
            ConvBlock(z_dim, z_dim)
        )

        self.init_to_rgb = WSConv2d(z_dim, img_channel, 1, 1, 0)

        self.upsampling = UpSampling()

        self.proj_layers = nn.ModuleList()
        self.up_to_rgb_layers = nn.ModuleList()
        self.proj_to_rgb_layers = nn.ModuleList()
        for i in range(len(in_channels_lst)):
            self.proj_layers.append(ConvBlock(in_channels_lst[i], out_channels_lst[i]))
            self.up_to_rgb_layers.append(WSConv2d(in_channels_lst[i], 3, 1, 1, 0))
            self.proj_to_rgb_layers.append(WSConv2d(out_channels_lst[i], 3, 1, 1, 0))

    def fade_in(self, alpha, upsampled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upsampled)

    # step: run through how many conv blocks
    # alpha: the fade in param
    def forward(self, x, alpha, step):
        x = self.init_conv(x)
        if step == -1:
            return self.init_to_rgb(x)

        generated = x
        for i in range(step + 1):
            upsampled = self.upsampling(generated)
            generated = self.proj_layers[i](upsampled)
        upsampled = self.up_to_rgb_layers[step](upsampled)
        generated = self.proj_to_rgb_layers[step](generated)
        return self.fade_in(alpha, upsampled, generated)


down_from_rgb_channels = [256, 256, 256, 256, 128, 64]
proj_from_rgb_channels = [256, 256, 256, 128, 64, 32]
class Disc(nn.Module):
    def __init__(self, img_channel=3) -> None:
        super(Disc, self).__init__()
        self.final_conv = nn.Sequential(
            ConvBlock(256, 256),
            nn.Conv2d(256, 256, 4, 1, 0)
        )

        self.init_from_rgb = WSConv2d(img_channel, 256, 1, 1, 0)

        self.proj_layers = nn.ModuleList()
        self.proj_from_rgb_layers = nn.ModuleList()
        self.down_from_rgb_layers = nn.ModuleList()
        for i in range(len(in_channels_lst)):
            self.proj_layers.append(ConvBlock(out_channels_lst[i], in_channels_lst[i]))
            self.down_from_rgb_layers.append(WSConv2d(img_channel, down_from_rgb_channels[i]))
            self.proj_from_rgb_layers.append(WSConv2d(img_channel, proj_from_rgb_channels[i]))
        self.down_sample = nn.AvgPool2d(2, 2)

    def fade_in(self, alpha, downsampled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * downsampled)

    def forward(self, x, alpha, step):
        if step == -1:
            x = self.init_from_rgb(x)
            return self.final_conv(x)

        downsampled = self.down_sample(x)
        downsampled = self.down_from_rgb_layers[step](downsampled)
        generated = self.proj_from_rgb_layers[step](x)
        generated = self.proj_layers[step](generated)
        generated = self.down_sample(generated)
        x = self.fade_in(alpha, downsampled, generated)

        downsampled = x
        for i in range(step - 1, -1, -1):
            projed = self.proj_layers[i](downsampled)
            downsampled = self.down_sample(projed)
        return self.final_conv(downsampled)




if __name__ == "__main__":
    model = Gen()
    x = torch.randn(8, 256, 1, 1)
    # print(Gen)
    print(model(x, 0.5, 5).shape)