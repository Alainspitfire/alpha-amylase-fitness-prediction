
# model_mutation_effect.py

import torch
import torch.nn as nn


class MutationEffectTransformer(nn.Module):
    def __init__(
        self,
        reference_tokens,
        vocab_size=22,
        max_len=425,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):
        super().__init__()

        reference_tokens = torch.as_tensor(
            reference_tokens, dtype=torch.long
        )

        if reference_tokens.ndim != 1 or len(reference_tokens) != max_len:
            raise ValueError("Reference length does not match max_len.")

        self.register_buffer(
            "reference_tokens",
            reference_tokens.clone()
        )

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=0
        )

        self.position_embedding = nn.Embedding(
            max_len, d_model
        )

        self.mutation_embedding = nn.Embedding(
            2, d_model
        )

        # NEW: learned representation of residue substitutions.
        self.delta_projection = nn.Linear(
            d_model, d_model, bias=False
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(2 * d_model)

        self.head = nn.Sequential(
            nn.Linear(2 * d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, mutation_mask=None):
        batch_size, length = x.shape

        if length != len(self.reference_tokens):
            raise ValueError(
                f"Expected {len(self.reference_tokens)} residues; "
                f"received {length}."
            )

        positions = torch.arange(
            length, device=x.device
        ).unsqueeze(0)

        reference = self.reference_tokens.unsqueeze(0)

        # Automatically construct the consensus-difference mask.
        computed_mask = (x != reference).long()

        if mutation_mask is not None:
            if not torch.equal(
                mutation_mask.long(), computed_mask
            ):
                raise ValueError(
                    "Mutation mask does not match the saved reference."
                )

        mutation_mask = computed_mask

        # Amino-acid representation of the observed sequence.
        mutant_emb = self.token_embedding(x)

        # Amino-acid representation of the consensus.
        reference_emb = self.token_embedding(
            self.reference_tokens
        ).unsqueeze(0)

        # NEW: explicit substitution embedding.
        delta_emb = mutant_emb - reference_emb

        delta_emb = self.delta_projection(delta_emb)

        # Combine all four sources of information.
        h = (
            mutant_emb
            + delta_emb
            + self.position_embedding(positions)
            + self.mutation_embedding(mutation_mask)
        )

        padding_mask = (x == 0)

        h = self.encoder(
            h,
            src_key_padding_mask=padding_mask
        )

        valid = (~padding_mask).unsqueeze(-1).float()

        # Global representation — unchanged.
        global_repr = (
            (h * valid).sum(dim=1)
            / valid.sum(dim=1).clamp(min=1)
        )

        # Mutation-aware mean pooling — unchanged.
        m = mutation_mask.unsqueeze(-1).float()

        mutation_repr = (
            (h * m).sum(dim=1)
            / m.sum(dim=1).clamp(min=1)
        )

        combined = torch.cat(
            [global_repr, mutation_repr],
            dim=-1
        )

        prediction = self.head(
            self.norm(combined)
        )

        return prediction.squeeze(-1)
