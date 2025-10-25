from main import initialize_model

return_dict = initialize_model('configs/test_config.yaml')
train_loader = return_dict['train_loader']

counter = 0
for batch_images, batch_labels, text_embd in train_loader:
    if counter >= 5:
        break

    print(f"Batch shape: {batch_images.shape}, Labels: {batch_labels}")

    print(f"Batch Embedding:")
    print(text_embd)

    print(f"Embedding Shape: {text_embd.shape}")

    counter += 1
