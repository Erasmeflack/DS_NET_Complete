# models/dual_snn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.siamese_branch import SiameseBranch
from models.classifier import MidLevelClassifier
from models.relation_module import RelationModule


class DualSiameseNet(nn.Module):
    """
    Dual-Siamese with runtime-controllable relation inputs (for ablations):
      - use_s_in_relation: include S-Net pairwise feature in relation module
      - use_d_in_relation: include D-Net pairwise feature in relation module
      - use_mid_in_relation: include mid (C-Net) flat feature in relation module

    Classification head is intentionally C-Net–only (mid map 256×4×4),
    to keep ablations clean for your A2/prototype-style experiments.
    """
    def __init__(
        self,
        freeze_top=False,
        freeze_mid=False,
        freeze_bottom=False,
        num_classes=5,
        agg_method="mean",
        device=None,
        # relation inputs (defaults = original behavior)
        use_s_in_relation: bool = True,
        use_d_in_relation: bool = True,
        use_mid_in_relation: bool = True,
    ):
        super(DualSiameseNet, self).__init__()

        # Branches
        self.top_branch = SiameseBranch()      # S-Net
        self.bottom_branch = SiameseBranch()   # D-Net
        self.mid_branch = SiameseBranch()      # C-Net feature extractor

        # Heads for S/D pairwise features
        self.similarity_fc    = nn.Linear(4096, 4096)
        self.similarity_final = nn.Linear(4096, 1)
        self.dissimilarity_fc = nn.Linear(4096, 4096)

        # Relation module expects 512×4×4 (we reduce channels before it)
        self.relation_module  = RelationModule(input_channels=512, spatial_size=4)
        self.channel_reducer  = nn.Conv2d(512 + 256, 512, kernel_size=1, bias=False)
        # ^ concatenates compact (512) pair-feature tensor with a 256-ch support map → reduce to 512

        # Classification head (C-Net only: 256×4×4)
        self.classifier = MidLevelClassifier(input_channels=256, spatial_size=4, num_classes=num_classes)

        self.num_classes = num_classes
        self.agg_method  = agg_method

        # Ablation toggles for relation inputs
        self.use_s_in_relation   = use_s_in_relation
        self.use_d_in_relation   = use_d_in_relation
        self.use_mid_in_relation = use_mid_in_relation

        # Freezing
        self._freeze_branch(self.top_branch, freeze_top)
        self._freeze_branch(self.mid_branch, freeze_mid)
        self._freeze_branch(self.bottom_branch, freeze_bottom)

        # Device
        if device is not None:
            self.to(device)
            self.device = device
        else:
            self.device = torch.device("cpu")

    # ---------------- Freezing helpers ----------------
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

    # ---------------- Utilities ----------------
    def set_relation_usage(self, use_s: bool = None, use_d: bool = None, use_mid: bool = None):
        """
        Runtime control to enable / disable S, D, Mid features going into Relation Module.
        Example: model.set_relation_usage(use_s=False)  # ablate S
        """
        if use_s is not None:
            self.use_s_in_relation = use_s
        if use_d is not None:
            self.use_d_in_relation = use_d
        if use_mid is not None:
            self.use_mid_in_relation = use_mid

    def _to_4d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 5 and x.size(0) == 1:
            x = x.squeeze(0)
        return x

    # ---------------- Pairwise mode (aux heads) ----------------
    def forward_pairwise(self, x1, x2):
        """
        Args:
            x1, x2: (B,1,64,64) or (1,1,64,64)
        Returns:
            sim_score: (B,) similarity prob (S-Net)
            dis_feat:  (B,4096) dissimilarity feature (D-Net)
            class_logits: (B,num_classes) C-Net-only classifier on x1
        """
        x1 = self._to_4d(x1).to(self.device)
        x2 = self._to_4d(x2).to(self.device)

        # S-Net similarity
        f_top_1_flat, f_top_2_flat, _ = self.top_branch(x1, x2)
        sim_feat  = torch.abs(f_top_1_flat - f_top_2_flat)           # (B,4096)
        sim_feat  = F.relu(self.similarity_fc(sim_feat))             # (B,4096)
        sim_score = torch.sigmoid(self.similarity_final(sim_feat)).squeeze(-1)  # (B,)

        # D-Net dissimilarity feature
        f_bot_1_flat, f_bot_2_flat, _ = self.bottom_branch(x1, x2)
        dis_feat = self.dissimilarity_fc(torch.abs(f_bot_1_flat - f_bot_2_flat))  # (B,4096)

        # C-Net-only classification on x1
        mid_map_x1 = self.mid_branch.encode(x1)                      # (B,256,4,4)
        class_logits = self.classifier(mid_map_x1)                   # (B,num_classes)

        return sim_score, dis_feat, class_logits

    # ---------------- Episodic mode (support-query) ----------------
    def forward(self, x_q, x_s):
        """
        Args:
            x_q: (Q,1,64,64) queries
            x_s: (S,1,64,64) supports
        Returns:
            relation_scores: (Q, S, 1)
            class_logits_q: (Q, num_classes)  — C-Net-only auxiliary
        """
        x_q = self._to_4d(x_q).to(self.device)
        x_s = self._to_4d(x_s).to(self.device)
        Q, S = x_q.size(0), x_s.size(0)

        # -------- Build relation input (runtime-controlled) --------
        parts = []

        if self.use_s_in_relation:
            f_top_q_flat, _, _ = self.top_branch(x_q, x_q)   # (Q,4096)
            f_top_s_flat, _, _ = self.top_branch(x_s, x_s)   # (S,4096)
            top_q = f_top_q_flat.unsqueeze(1).expand(Q, S, -1)
            top_s = f_top_s_flat.unsqueeze(0).expand(Q, S, -1)
            f_s = F.relu(self.similarity_fc(torch.abs(top_q - top_s).reshape(-1, 4096)))  # (Q*S,4096)
            parts.append(f_s)

        if self.use_d_in_relation:
            f_bot_q_flat, _, _ = self.bottom_branch(x_q, x_q)  # (Q,4096)
            f_bot_s_flat, _, _ = self.bottom_branch(x_s, x_s)  # (S,4096)
            bot_q = f_bot_q_flat.unsqueeze(1).expand(Q, S, -1)
            bot_s = f_bot_s_flat.unsqueeze(0).expand(Q, S, -1)
            f_d = F.relu(self.dissimilarity_fc(torch.abs(bot_q - bot_s).reshape(-1, 4096)))  # (Q*S,4096)
            parts.append(f_d)

        if self.use_mid_in_relation:
            f_mid_q_flat, _, _ = self.mid_branch(x_q, x_q)   # (Q,4096)
            mid_q = f_mid_q_flat.unsqueeze(1).expand(Q, S, -1).reshape(-1, 4096)            # (Q*S,4096)
            parts.append(mid_q)

        # Fallback guard: if all were disabled, still pass zeros to relation
        if len(parts) == 0:
            z_linear = torch.zeros(Q * S, 512, device=x_q.device)  # minimal safe tensor
        else:
            z_linear = torch.cat(parts, dim=1)  # (Q*S, 4096 * (#enabled))
            # compact to ~512, then shape to (512,4,4)
            k = max(1, z_linear.size(1) // 512)
            z_red = F.avg_pool1d(z_linear.unsqueeze(1), kernel_size=k, stride=k).squeeze(1)   # (Q*S, ~512)
            if z_red.size(1) != 512:
                # project or slice to 512
                eye = torch.eye(z_red.size(1), device=z_red.device)[:512]
                z_red = F.linear(z_red, eye)  # (Q*S, 512)
            z_red = z_red.view(-1, 512, 1, 1)
            z_red = F.interpolate(z_red, size=(4, 4), mode="bilinear", align_corners=False)   # (Q*S,512,4,4)

        # Inject a support spatial map (mid) to keep spatial context (always C-Net map; ablations are for relation vectors)
        mid_s_map = self.mid_branch.encode(x_s)  # (S,256,4,4)
        mid_s_map_rep = mid_s_map.unsqueeze(0).repeat(Q, 1, 1, 1, 1).view(-1, 256, 4, 4)  # (Q*S,256,4,4)

        # Concatenate and reduce to 512 for relation module
        z = torch.cat([z_red if len(parts) > 0 else torch.zeros_like(z_red), mid_s_map_rep], dim=1)  # (Q*S, 512+256,4,4)
        z = self.channel_reducer(z)  # (Q*S,512,4,4)

        # Relation scores
        relation_scores = self.relation_module(z).view(Q, S, 1)  # (Q,S,1)

        # Auxiliary C-Net-only logits for queries
        mid_q_map = self.mid_branch.encode(x_q)                  # (Q,256,4,4)
        class_logits_q = self.classifier(mid_q_map)              # (Q,num_classes)

        return relation_scores, class_logits_q

    # ---------------- Prediction from relations ----------------
    def predict_from_relations(self, relation_scores, support_labels):
        """
        relation_scores: (Q,S,1) or (Q,S)
        support_labels:  (S,) or (1,S)
        returns: (Q,)
        """
        if relation_scores.dim() == 3 and relation_scores.size(-1) == 1:
            relation_scores = relation_scores.squeeze(-1)    # (Q, S)
        support_labels = support_labels.view(-1)             # (S,)

        Q, S = relation_scores.shape
        out = torch.zeros(Q, self.num_classes, device=relation_scores.device)

        for c in range(self.num_classes):
            mask = (support_labels == c)
            if mask.any():
                per_class = relation_scores[:, mask]         # (Q, S_c)
                vals = per_class.max(dim=1).values if self.agg_method == "max" else per_class.mean(dim=1)
                out[:, c] = vals

        return out.argmax(dim=1)
