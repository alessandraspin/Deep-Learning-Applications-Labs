from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

def zero_shot_classification(dataset_split, model, processor, device):
    class_labels = dataset_split.features['label'].names
    text_prompts = [f"an image of a {label}" for label in class_labels]
    
    # Processing of texts
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    
    correct_predictions = 0
    total_images = len(dataset_split)

    for example in tqdm(dataset_split, desc="Evaluating Zero-Shot Performance"):
        image = example['image']

        # Processing of images
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Embeddings
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)

        # Normalization
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T)
        probs = similarity.softmax(dim=-1)
        
        prediction = torch.argmax(probs).item()
        
        if prediction == example['label']:
            correct_predictions += 1
            
    accuracy = correct_predictions / total_images

    return accuracy

def zero_shot_classification_with_plots(dataset_split, model, processor, device, id2label, n_plot=15):
    class_labels = list(id2label.values())
    text_prompts = [f"an image of a {label}" for label in class_labels]

    # Processing of texts
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    correct_predictions = 0
    total_images = len(dataset_split)

    # per plotting
    examples_for_plot = []

    for i, example in enumerate(tqdm(dataset_split, desc="Evaluating Zero-Shot Performance")):
        image = example['image']
        true_label_id = example['label']

        # Processing of images
        image_inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)

        # Normalization
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T)
        probs = similarity.softmax(dim=-1)

        prediction = torch.argmax(probs).item()

        if prediction == true_label_id:
            correct_predictions += 1

        # Examples for plotting
        if len(examples_for_plot) < n_plot:
            examples_for_plot.append({
                "image": image,
                "true_label": id2label[true_label_id],
                "pred_label": id2label[prediction],
                "correct": prediction == true_label_id
            })

    accuracy = correct_predictions / total_images

    # Plot
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for ax, ex in zip(axes.flat, examples_for_plot):
        ax.imshow(ex["image"], cmap="gray")
        color = "green" if ex["correct"] else "red"
        ax.set_title(f"True: {ex['true_label']}\nPred: {ex['pred_label']}", 
                     fontsize=16, color=color)
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()

    return accuracy


