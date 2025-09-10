import torch
import torch.nn as nn
from torch import Tensor
import timm
from timm.utils import freeze


class ConvNeXt(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1,
        freeze_stem: bool = True,
    ) -> None:
        super(ConvNeXt, self).__init__()
            
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
        )
        
#        print("Model:", self.model)

        if freeze_stem:
            freeze(self, 'model.stem')

#        data_config = timm.data.resolve_model_data_config(self.model)
#        print("data_config:", data_config)

    def forward(self, x) -> Tensor:
        y = self.model(x)               # [N, num_classes]

        return y

if __name__ == "__main__":
    import torch
    
    model = ConvNext(model_name="convnext_base.fb_in22k_ft_in1k_384",
                      pretrained=True,
                      in_channels=4,
                      num_classes=3,
                      freeze_stem=True)
    
    x = torch.rand((5, 4, 384, 384), dtype=torch.float32)
    y, f = model(x)
    print("y:", y.shape, f.shape)
