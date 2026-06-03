What's done:
- DataLoader

Next to do:
- Set up Model config for IFT - done
- train loop



The Train Loop
For each epoch:
    For each batch:
        logits = model(batch)
        loss = cross_entropy(logits, batch.labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
