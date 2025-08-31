# models/dual_snn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.siamese_branch import SiameseBranch
from models.classifier import MidLevelClassifier
from models.relation_module import RelationModule


class DualSiameseNet(nn.Module):
    def __init__(
        self,
        freeze_top=False,
        freeze_mid=False,
        freeze_bottom=False,
        num_classes=5,
        agg_method="mean",
        device=None,
    ):
        super(DualSiameseNet, self).__init__()

        # Siamese sub-networks
        self.top_branch = SiameseBranch()      # S-Net
        self.bottom_branch = SiameseBranch()   # D-Net
        self.mid_branch = SiameseBranch()      # C-Net feature extractor

        # Heads for pairwise features
        self.similarity_fc   = nn.Linear(4096, 4096)
        self.similarity_final= nn.Linear(4096, 1)
        self.dissimilarity_fc= nn.Linear(4096, 4096)

        # Classification head over concatenated maps: (256*3, 4, 4)
        self.classifier = MidLevelClassifier(input_channels=768, spatial_size=4, num_classes=num_classes)

        # Relation module (expects 512×4×4 after channel reduction)
        self.relation_module = RelationModule(input_channels=512, spatial_size=4)
        self.channel_reducer = nn.Conv2d(768, 512, kernel_size=1, bias=False)

        self.num_classes = num_classes
        self.agg_method  = agg_method

        # Freeze options
        self._freeze_branch(self.top_branch, freeze_top)
        self._freeze_branch(self.mid_branch, freeze_mid)
        self._freeze_branch(self.bottom_branch, freeze_bottom)

        # Device
        if device is not None:
            self.to(device)
            self.device = device
        else:
            self.device = torch.device("cpu")

    # ---------- Freezing helpers ----------
    def freeze_similarity_head(self, freeze: bool):
        for p in self.similarity_fc.parameters():
            p.requires_grad = not freeze
        for p in self.similarity_final.parameters():
            p.requires_grad = not freeze

    def freeze_dissimilarity_head(self, freeze: bool):
        for p in self.dissimilarity_fc.parameters():
            p.requires_grad = not freeze

    def freeze_classifier(self, freeze: bool):
        for p in self.classifier.parameters():
            p.requires_grad = not freeze

    def _freeze_branch(self, branch, freeze):
        if freeze:
            for p in branch.parameters():
                p.requires_grad = False

    # ---------- Utilities ----------
    def _concat_query_maps_768(self, x_q):
        """
        Build 768-ch query feature map by concatenating top/bottom/mid maps along channels.
        Assumes each encode() -> (B, 256, 4, 4).
        """
        top_q_map = self.top_branch.encode(x_q)     # (B,256,4,4)
        bot_q_map = self.bottom_branch.encode(x_q)  # (B,256,4,4)
        mid_q_map = self.mid_branch.encode(x_q)     # (B,256,4,4)
        merged_q  = torch.cat([top_q_map, bot_q_map, mid_q_map], dim=1)  # (B,768,4,4)
        return merged_q

    def _to_4d(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [C,H,W] -> [1,C,H,W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # Accept [1, B, C, H, W] -> [B, C, H, W]
        if x.dim() == 5 and x.size(0) == 1:
            x = x.squeeze(0)
        # Sanity: must be 4D now
        return x
    
    # ---------- Pairwise mode (S/D/C aux training) ----------
    def forward_pairwise(self, x1, x2):
        """
        Args:
            x1, x2: (B,1,64,64) or (1,1,64,64)
        Returns:
            sim_score: scalar prob (B,) from S-Net
            dis_feat:  (B,4096) from D-Net
            class_logits: (B,num_classes) from C-Net (using fused 768-ch query map of x1)
        """
        x1 = self._to_4d(x1).to(self.device)
        x2 = self._to_4d(x2).to(self.device)
        if x1.dim() == 3: x1 = x1.unsqueeze(0)
        if x2.dim() == 3: x2 = x2.unsqueeze(0)
        x1, x2 = x1.to(self.device), x2.to(self.device)

        # S-Net similarity (pairwise)
        f_top_1_flat, f_top_2_flat, _ = self.top_branch(x1, x2)
        sim_feat  = torch.abs(f_top_1_flat - f_top_2_flat)       # (B,4096)
        sim_feat  = F.relu(self.similarity_fc(sim_feat))         # (B,4096)
        sim_score = torch.sigmoid(self.similarity_final(sim_feat)).squeeze(-1)  # (B,)

        # D-Net dissimilarity features (pairwise)
        f_bot_1_flat, f_bot_2_flat, _ = self.bottom_branch(x1, x2)
        dis_feat = self.dissimilarity_fc(torch.abs(f_bot_1_flat - f_bot_2_flat))  # (B,4096)

        # C-Net classification on fused query map (x1 as "query")
        merged_q = self._concat_query_maps_768(x1)               # (B,768,4,4)
        class_logits = self.classifier(merged_q)                 # (B,num_classes)

        return sim_score, dis_feat, class_logits

    # ---------- Episodic mode (support-query) ----------
    def forward(self, x_q, x_s):
        """
        Args:
            x_q: (B_q,1,64,64), query batch
            x_s: (B_s,1,64,64), support batch
        Returns:
            relation_scores: (B_q, B_s, 1)
            class_logits_q: (B_q, num_classes)  — auxiliary
        """
        x_q = self._to_4d(x_q).to(self.device)  # [B_q,1,64,64]
        x_s = self._to_4d(x_s).to(self.device)  # [B_s,1,64,64]
        x_q, x_s = x_q.to(self.device), x_s.to(self.device)
        B_q, B_s = x_q.size(0), x_s.size(0)

        # Flat features for S/D pairwise diffs
        f_top_q_flat, _, _ = self.top_branch(x_q, x_q)   # (B_q,4096)
        f_top_s_flat, _, _ = self.top_branch(x_s, x_s)   # (B_s,4096)
        f_bot_q_flat, _, _ = self.bottom_branch(x_q, x_q)# (B_q,4096)
        f_bot_s_flat, _, _ = self.bottom_branch(x_s, x_s)# (B_s,4096)
        f_mid_q_flat, _, _ = self.mid_branch(x_q, x_q)   # (B_q,4096)
        f_mid_s_flat, _, _ = self.mid_branch(x_s, x_s)   # (B_s,4096)

        # Expand to pairwise grid
        top_q = f_top_q_flat.unsqueeze(1).expand(B_q, B_s, -1)   # (B_q,B_s,4096)
        top_s = f_top_s_flat.unsqueeze(0).expand(B_q, B_s, -1)   # (B_q,B_s,4096)
        bot_q = f_bot_q_flat.unsqueeze(1).expand(B_q, B_s, -1)
        bot_s = f_bot_s_flat.unsqueeze(0).expand(B_q, B_s, -1)

        # S-branch similarity feature
        f_s = F.relu(self.similarity_fc(torch.abs(top_q - top_s).reshape(-1, 4096)))  # (B_q*B_s,4096)
        # D-branch dissimilarity feature
        f_d = F.relu(self.dissimilarity_fc(torch.abs(bot_q - bot_s).reshape(-1, 4096)))  # (B_q*B_s,4096)

        # Mid flat (optional if you want it in relation fusion)
        mid_q = f_mid_q_flat.unsqueeze(1).expand(B_q, B_s, -1).reshape(-1, 4096)  # (B_q*B_s,4096)

        # Build a compact 512×4×4 tensor for relation module
        # Concatenate [f_s, f_d, mid_q] -> (B_q*B_s, 12288), shrink to 512 via avg-pool then upsample to 4×4
        z_linear = torch.cat([f_s, f_d, mid_q], dim=1)  # (B_q*B_s, 12288)
        # Kernel size chosen to map 12288 → 512 exactly (12288/24=512). Safe guard if dims change:
        k = max(1, z_linear.size(1) // 512)
        z_red = F.avg_pool1d(z_linear.unsqueeze(1), kernel_size=k, stride=k).squeeze(1)  # (B_q*B_s, ~512)
        # If not exact 512, project to 512
        if z_red.size(1) != 512:
            z_red = F.linear(z_red, torch.eye(z_red.size(1), device=z_red.device)[:512])  # slice/proj
        z_red = z_red.view(-1, 512, 1, 1)
        z_red = F.interpolate(z_red, size=(4, 4), mode="bilinear", align_corners=False)  # (B_q*B_s,512,4,4)

        # Also inject a support spatial map (mid) to keep spatial context
        mid_s_map = self.mid_branch.encode(x_s)  # (B_s,256,4,4)
        mid_s_map_rep = mid_s_map.unsqueeze(0).repeat(B_q, 1, 1, 1, 1).view(-1, 256, 4, 4)  # (B_q*B_s,256,4,4)

        # Concatenate -> 768 then reduce to 512 for relation module
        z = torch.cat([z_red, mid_s_map_rep], dim=1)  # (B_q*B_s,768,4,4)
        z = self.channel_reducer(z)                   # (B_q*B_s,512,4,4)

        # Relation score per (q,s)
        relation_scores = self.relation_module(z)     # (B_q*B_s,1)
        relation_scores = relation_scores.view(B_q, B_s, 1)

        # Auxiliary classification for query via fused 768-ch map
        merged_q = self._concat_query_maps_768(x_q)   # (B_q,768,4,4)
        class_logits_q = self.classifier(merged_q)    # (B_q,num_classes)

        return relation_scores, class_logits_q

    # ---------- Prediction from relations ----------
    def predict_from_relations(self, relation_scores, support_labels):
        """
        relation_scores: (Q, S, 1) or (Q, S)
        support_labels:  (S,) or (1, S)
        returns: (Q,)
        """
        if relation_scores.dim() == 3 and relation_scores.size(-1) == 1:
            relation_scores = relation_scores.squeeze(-1)    # (Q, S)
        support_labels = support_labels.view(-1)             # (S,)

        Q, S = relation_scores.shape
        out = torch.zeros(Q, self.num_classes, device=relation_scores.device)

        for c in range(self.num_classes):
            mask = (support_labels == c)                     # (S,)
            if mask.any():
                per_class = relation_scores[:, mask]         # (Q, S_c)
                if self.agg_method == "max":
                    vals, _ = per_class.max(dim=1)
                else:
                    vals = per_class.mean(dim=1)
                out[:, c] = vals

        return out.argmax(dim=1)
