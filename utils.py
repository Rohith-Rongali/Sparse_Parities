import torch

def generate_data(data_config):
    # n = 128
    # k = 7
    num_points = data_config.num_points
    n = data_config.n
    k = data_config.k
    alpha = data_config.alpha
    bias = data_config.bias

    data = torch.zeros(num_points, n)
    
    # Generate alpha fraction of points with {-1, 1} coordinates
    alpha_points = int(num_points * alpha)
    if alpha > 0.0:
        probabilities = torch.tensor([1-bias, bias])
        alpha_data = torch.multinomial(probabilities, alpha_points * n, replacement=True).view(alpha_points, n).type(torch.FloatTensor)
        alpha_data = alpha_data * 2 - 1
        data[:alpha_points] = alpha_data
    
    # Generate (1-alpha) fraction of points with {-1, 1} coordinates
    uniform_points = num_points - alpha_points
    uniform_data = torch.randint(0, 2, size=(uniform_points, n)).type(torch.FloatTensor)
    uniform_data = uniform_data * 2 - 1
    data[alpha_points:] = uniform_data

    indices = torch.randint(0, n, (k,))
    selected_columns = data.index_select(1, indices)
    labels = torch.prod(selected_columns, dim=1)
    
    return data,labels

# Train model
def train(model,optimizer,loss_fn,num_epochs,train_loader,data,device):
    
    x_train, x_test, y_train, y_test = data
    model.train()
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = model(data.to(device))
            loss = loss_fn(output, target.to(device))  # Unsqueeze target to match output shape
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            # if batch_idx % 10 == 0:
        train_output = model(x_train.to(device))
        train_loss.append(loss_fn(train_output, y_train.to(device)).item())
        test_output = model(x_test.to(device))
        test_loss.append(loss_fn(test_output, y_test.to(device)).item())
        print(f"Epoch: {epoch} | Train Loss: {train_loss[-1]} | Test Loss: {test_loss[-1]}")
        if train_loss[-1] < 1e-5:
            break
    return train_loss, test_loss