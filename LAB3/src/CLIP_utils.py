import torch
import torch.nn as nn

class CLIPForClassification(nn.Module):
    def __init__(self, clip_model, num_labels):
        super().__init__()
        self.clip = clip_model
        self.num_labels = num_labels
        # testuale: class embeddings
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_labels, bias=False)

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
        # Otteniamo feature immagine
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output  # [batch, hidden_dim]

        # Proiettiamo nello spazio CLIP
        image_embeds = self.clip.visual_projection(pooled)

        # Classificazione
        logits = self.classifier(image_embeds)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}