# Implementation of Image Background Serves as Good Proxy for Out-of-distribution Data
# https://arxiv.org/abs/2307.00519

import argparse
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import einops

class LinearClassifier(nn.Module):
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x, inputmap=False):
        if inputmap:
            x = einops.rearrange(x, 'B c h w -> (B h w) c')
        else:
            # flatten
            x = x.view(x.size(0), -1)

        return self.linear(x)

def forward_map(model, x):
    # architecture of model is resnet50
    # return the avgpool & feature 
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    return model.avgpool(x), x

def train(cls_head, ood_head, model, train_loader, args):

    optimizer = Adam(
        list(cls_head.parameters()) + list(model.parameters()) + list(ood_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    for epoch in range(args.epochs):
        cls_head.train()
        ood_head.train()
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            loss = 0
            images, class_labels = batch
            images = images.to(device, non_blocking=True) # [B, 3, 224, 224]
            class_labels = class_labels.to(device, non_blocking=True) # [B]

            f_gap, f_map = forward_map(model, images)
            logits_gap = cls_head(f_gap)

            # cross entropy loss
            ce_crit = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
            ce_loss = ce_crit(logits_gap, class_labels)
            loss += ce_loss

            with torch.no_grad():
                B, c, h, w = f_map.shape
                assert h==7
                assert w==7
                assert c==2048
                logits_map = cls_head(f_map, inputmap=True) # [B*h*w, class]
                conf_map = nn.Softmax(dim=1)(logits_map)
                conf_map_target = torch.gather(
                    conf_map, 
                    dim=1, 
                    index=class_labels.repeat_interleave(h*w).unsqueeze(-1)
                )

                mask_id = (conf_map_target >= args.thres)
                mask_ood = (conf_map_target <= 1 - args.thres)
            
            # (binary) cross entropy loss
            conf_map_pred = ood_head(f_map, inputmap=True)
            ce_crit = nn.CrossEntropyLoss(reduction='none')
            # Loss-Wise Balance
            id_logits = conf_map_pred[mask_id.repeat(1, 2)].reshape(-1, 2) # [B*h*w, 2]
            ood_logits = conf_map_pred[mask_ood.repeat(1, 2)].reshape(-1, 2)
            id_targets = torch.ones(mask_id.sum(), dtype=int).cuda()
            ood_targets = torch.zeros(mask_ood.sum(), dtype=int).cuda()
            bce_id_loss = ce_crit(id_logits, id_targets).mean()
            bce_ood_loss = ce_crit(ood_logits, ood_targets).mean()
            loss += (bce_id_loss + bce_ood_loss) * args.bce_weight
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Step schedule
        exp_lr_scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--base_model', type=str, default='resnet50_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--label_smooth', type=float, default=0.0)

    parser.add_argument('--thres', type=float, default=0.99)
    parser.add_argument('--bce_weight', type=float, default=1.0)
    parser.add_argument('--ood_head_type', type=str, default="Sig", help="Sig \ Sm \ Linear")
    args = parser.parse_args()

    device = torch.device('cuda:0')
    args.num_labeled_classes = 1000

    # ----------------------
    # BACKBONE
    # ----------------------
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    args.image_size = 224
    args.feat_dim = 2048
    model.to(device)

    # ----------------------
    # CLS HEAD & OOD Head
    # ----------------------
    cls_head = LinearClassifier(args.feat_dim, num_labels=args.num_labeled_classes)
    ood_head = LinearClassifier(args.feat_dim, num_labels=2)
    cls_head.to(device)
    ood_head.to(device)

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = None

    # ----------------------
    # TRAIN
    # ----------------------
    train(cls_head, ood_head, model, train_loader, args)
