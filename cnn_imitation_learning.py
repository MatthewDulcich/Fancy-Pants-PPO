import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, input_channels=1):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the output from the convolutional layers
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU()
        )
        self._get_conv_output_size(input_channels)
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 5)  # Assuming 5 possible actions

    def _get_conv_output_size(self, input_channels):
        # Pass a dummy input through the convolutional layers to get the output size
        with torch.no_grad():
            x = torch.randn(1, input_channels, 150, 200)
            x = self.convs(x)
            self._to_linear = x.numel()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.convs(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_action(self, state):
        state = state.unsqueeze(0)  # Add the batch dimension
        action_logits = self.forward(state)
        action_prob = torch.softmax(action_logits, dim=1)
        action = torch.argmax(action_prob, dim=1)
        return action.item(), action_prob

    def calculate_entropy(self):
        dummy_input = torch.randn(1, 150, 200)  # Dummy input without channel dimension
        action_logits = self.forward(dummy_input)
        action_prob = torch.softmax(action_logits, dim=1)  # Action probabilities
        return -(action_prob * torch.log(action_prob + 1e-8)).sum(dim=1)  # Entropy calculation

if __name__ == '__main__':
    actor = Actor(input_channels=1)
    state = torch.randn(150, 200)  # Random state with shape (height, width)
    # Print the output action value of the actor
    action, action_prob = actor.get_action(state)
    print(f"Action: {action}, Action Probabilities: {action_prob}")

    # Hyperparameters
    learning_rate = 0.1
    num_epochs = 2

    # Dummy dataset (replace with your actual dataset)
    frames = torch.randn(5, 150, 200)  # 5 frames with shape (batch_size, height, width)
    actions = torch.normal(mean=2, std=1, size=(5,)).clamp(0, 4).long()  # 5 actions with a normal distribution, clamped between 0 and 4

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(actor.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for i in tqdm(range(len(frames)), desc="Training Epoch"):
            frame = frames[i].unsqueeze(0)  # Add batch dimension
            action = actions[i]

            # Forward pass
            action_logits = actor.forward(frame)
            loss = criterion(action_logits, action.unsqueeze(0))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(frames)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"Action probabilities for epoch {epoch+1}:")
        for i in range(len(frames)):
            frame = frames[i].unsqueeze(0)  # Add batch dimension
            _, action_prob = actor.get_action(frame.squeeze(0))
            print(f"Frame {i+1}: {action_prob}")
            
        # Save the model
        model_path = 'actor_model.pth'
        torch.save(actor.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Code to load the model (commented out for future use)
        # actor = Actor(input_channels=1)
        # actor.load_state_dict(torch.load(model_path))
        # actor.eval()