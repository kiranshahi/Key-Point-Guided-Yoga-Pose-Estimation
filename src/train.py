import copy

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

best_val_loss = float("inf")
best_weights = None

for epoch in range(30):
    model.train()

    running_loss = 0

    for images, keypoints, mask, labels in train_loader:
        images = images.to(device)
        keypoints = keypoints.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss, cls_loss, pose_loss = multitask_loss(outputs, keypoints, mask, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(images)

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, keypoints, mask, labels in val_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss, _, _ = multitask_loss(outputs, keypoints, mask, labels)

            val_loss += loss.item() * len(images)
            correct += (outputs["classes"].argmax(1) == labels).sum().item()
            total += len(labels)

    train_avg = running_loss / len(train_data)
    val_avg = val_loss / len(val_data)
    accuracy = correct / total

    print(f"Epoch {epoch+1:02d} | "f"Train Loss: {train_avg:.4f} | "f"Val Loss: {val_avg:.4f} | "f"Val Accuracy: {accuracy:.2%}")

    if val_avg < best_val_loss:
        best_val_loss = val_avg
        best_weights = copy.deepcopy(model.state_dict())

model.load_state_dict(best_weights)

torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES, "image_size": 224}, "/kaggle/working/yoga_multitask.pt")