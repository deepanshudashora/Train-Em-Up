from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models.neural_nets import SimpleConvNet
import json
from threading import Thread, Event, Lock
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import time
from torch.cuda import is_available
import psutil
import math
import base64
import io
from PIL import Image
import torch.nn.functional as F

app = Flask(__name__)

# Global variables to store training state
training_logs = {
    'model1': {
        'loss': [], 'accuracy': [], 
        'test_loss': [], 'test_accuracy': [],
        'log_messages': []
    },
    'model2': {
        'loss': [], 'accuracy': [], 
        'test_loss': [], 'test_accuracy': [],
        'log_messages': []
    }
}
current_dataset = None
models = {'model1': None, 'model2': None}

# Add these global variables
training_progress = {
    'model1': {'current_epoch': 0, 'progress': 0},
    'model2': {'current_epoch': 0, 'progress': 0}
}

# Update these global variables
AVAILABLE_MEMORY = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # in GB
MAX_TOTAL_MEMORY = AVAILABLE_MEMORY * 0.7  # 70% of available memory
MAX_MODEL_MEMORY = MAX_TOTAL_MEMORY / 2  # 35% per model

# Add these at the top with other globals
progress_lock = Lock()
start_event = Event()

# def find_nearest_power2_combination(test_data_size, user_input_batch_size):
#     # Dynamically calculate popular batch sizes up to the test_data_size
#     popular_batch_sizes = []
#     size = 4
#     while size <= test_data_size:
#         popular_batch_sizes.append(size)
#         size *= 2  # Double the size to get the next popular batch size

#     # Initialize closest batch size to zero
#     closest_batch_size = 0

#     # Find the closest valid batch size
#     for size in popular_batch_sizes:
#         if size >= user_input_batch_size and size <= test_data_size:
#             closest_batch_size = size
#             break

#     # If no size is found in the loop, it means test_data_size is smaller than the smallest batch size
#     if closest_batch_size == 0:
#         closest_batch_size = min(popular_batch_sizes, key=lambda x: abs(x - user_input_batch_size))
        
#         # Ensure it does not exceed the maximum allowed size
#         if closest_batch_size > test_data_size:
#             closest_batch_size = test_data_size
    
#     return closest_batch_size



def calculate_max_memory_based_batch_size(image_size=28 * 28, channels=1, dtype_size=4):
    """
    Calculate the maximum allowed batch size based on 70% of system RAM.
    
    Parameters:
        image_size (int): Size of the image in pixels (height * width).
        channels (int): Number of color channels (e.g., 1 for grayscale, 3 for RGB).
        dtype_size (int): Number of bytes per data type (e.g., 4 for float32).
        
    Returns:
        int: Maximum batch size that can fit in 70% of system RAM.
    """
    # Calculate memory required per sample (image)
    bytes_per_sample = image_size * channels * dtype_size
    
    # Calculate 70% of total system RAM in bytes
    total_ram = psutil.virtual_memory().total
    memory_limit = int(0.7 * total_ram)  # 70% of system RAM in bytes

    # Calculate the maximum batch size that fits within 70% of system RAM
    memory_based_max = int(memory_limit / bytes_per_sample)
    
    return memory_based_max


def closest_valid_batch_size(test_data_size, memory_based_max, user_input_batch_size):
    # Dynamically calculate popular batch sizes up to the minimum of test_data_size and memory_based_max
    popular_batch_sizes = []
    size = 4
    max_limit = min(test_data_size, memory_based_max)  # Limit based on both test data size and memory constraint
    while size <= max_limit:
        popular_batch_sizes.append(size)
        size *= 2  # Double the size to get the next popular batch size

    # Initialize closest batch size to zero
    closest_batch_size = 0

    # Find the closest valid batch size that meets or exceeds user_input_batch_size
    for size in popular_batch_sizes:
        if size >= user_input_batch_size:
            closest_batch_size = size
            break

    # If no suitable size is found, choose the largest available batch size within limits
    if closest_batch_size == 0:
        closest_batch_size = popular_batch_sizes[-1] if popular_batch_sizes else 0
    
    return closest_batch_size


def get_dataset_sizes(train_dataset, test_dataset):
    """Get valid batch size based on dataset sizes"""
    train_size = len(train_dataset)
    test_size = len(test_dataset)
       
    # Use the smaller of the two
    return min(train_size, test_size)

def calculate_max_batch_size(train_dataset, test_dataset, user_input_batch_size, image_size=28*28):
    memory_based_max = calculate_max_memory_based_batch_size()
    
    # Get valid dataset size
    dataset_min = get_dataset_sizes(train_dataset, test_dataset)
    
    # First limit by dataset size and memory
    max_batch = closest_valid_batch_size(dataset_min, memory_based_max, user_input_batch_size)
        
    print("\nBatch Size Estimation:")
    print(f"├─ Valid dataset size: {dataset_min}")
    print(f"├─ Memory-based maximum: {memory_based_max}")
    print(f"└─ Valid batch size: {max_batch}")
    
    return max_batch

def prepare_loaders(train_dataset, test_dataset, user_input_batch_size):
    """Prepare data loaders with proper batch size handling and accumulation steps if needed"""
    # Calculate maximum possible batch size considering memory and dataset size constraints
    max_possible_batch = calculate_max_batch_size(train_dataset, test_dataset, user_input_batch_size)
    
    # Determine accumulation steps
    if user_input_batch_size > max_possible_batch:
        accumulation_steps = math.ceil(user_input_batch_size / max_possible_batch)
    else:
        accumulation_steps = 1  # No accumulation needed if user batch size fits in memory
    
    print(f"Accumulation Steps: {accumulation_steps}") 
    print("\nBatch Configuration:")
    print(f"├─ User requested: {user_input_batch_size}")
    print(f"├─ Valid batch: {max_possible_batch}")
    print(f"├─ Accumulation steps: {accumulation_steps}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=max_possible_batch,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=max_possible_batch,
        shuffle=False
    )
    
    return train_loader, test_loader, max_possible_batch, accumulation_steps

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/select_dataset', methods=['POST'])
def select_dataset():
    dataset_type = request.form['dataset']
    global current_dataset
    current_dataset = dataset_type
    return jsonify({'redirect': '/hyperparameters'})

@app.route('/hyperparameters')
def hyperparameters():
    return render_template('hyperparameters.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    params = request.get_json()
    
    # Separate model parameters from training parameters
    def get_model_params(config):
        return {
            'channels': config['channels'],
            'dropout_rate': config['dropout_rate']
        }
    
    def get_training_params(config):
        return {
            'batch_size': int(config['batch_size']),
            'learning_rate': float(config['learning_rate']),
            'optimizer': config['optimizer'],
            'epochs': int(config['epochs'])  # Add epochs to training params
        }
    
    # Create models with only relevant parameters
    models['model1'] = SimpleConvNet(**get_model_params(params['model1']))
    models['model2'] = SimpleConvNet(**get_model_params(params['model2']))
    
    # Store training parameters
    training_params = {
        'model1': get_training_params(params['model1']),
        'model2': get_training_params(params['model2'])
    }
    
    # Clear previous training logs
    training_logs['model1']['log_messages'] = []
    training_logs['model2']['log_messages'] = []
    training_logs['model1']['loss'] = []
    training_logs['model1']['accuracy'] = []
    training_logs['model2']['loss'] = []
    training_logs['model2']['accuracy'] = []
    
    # Start training in background thread
    thread = Thread(target=train_models, args=(training_params,))
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/get_training_status')
def get_training_status():
    print("Training Status Data:", training_logs)  # Debug print
    return jsonify(training_logs)

def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if current_dataset == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    else:
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
    
    return train_dataset, test_dataset

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def train_models(params):
    train_dataset, test_dataset = load_dataset()
    
    # Initialize progress for both models before starting training
    for model_name in ['model1', 'model2']:
        training_progress[model_name] = {
            'current_epoch': 0,
            'total_epochs': int(params[model_name]['epochs']),
            'progress': 0,
            'current_loss': 0,
            'current_accuracy': 0,
            'current_test_loss': 0,
            'current_test_accuracy': 0,
            'batch_size': 0,
            'effective_batch_size': 0
        }
    
    # Prepare loaders for each model
    loaders = {}
    for model_name in ['model1', 'model2']:
        original_batch_size = int(params[model_name]['batch_size'])
        train_loader, test_loader, actual_batch, accumulation_steps = prepare_loaders(
            train_dataset, 
            test_dataset, 
            original_batch_size
        )
        
        # Update progress with batch size information
        training_progress[model_name].update({
            'batch_size': actual_batch,
            'effective_batch_size': actual_batch * accumulation_steps,
            'original_batch_size': original_batch_size
        })
        
        loaders[model_name] = {
            'train': train_loader,
            'test': test_loader,
            'batch_size': actual_batch,
            'accumulation_steps': accumulation_steps
        }
        
        params[model_name].update({
            'batch_size': actual_batch,
            'accumulation_steps': accumulation_steps
        })
    
    # Reset the start event
    start_event.clear()
    
    # Start training threads
    threads = []
    for model_name in ['model1', 'model2']:
        thread = Thread(target=train_single_model,
                       args=(models[model_name], 
                             get_optimizer(models[model_name], params[model_name]),
                             loaders[model_name]['train'],
                             loaders[model_name]['test'],
                             model_name,
                             params[model_name]))
        thread.start()
        threads.append(thread)
    
    # Signal both threads to start training
    start_event.set()
    
    # Wait for completion
    for thread in threads:
        thread.join()

def train_single_model(model, optimizer, train_loader, test_loader, model_name, config):
    criterion = nn.CrossEntropyLoss()
    n_epochs = config['epochs']
    original_batch_size = config['batch_size']
    
    # Get proper batch sizes and accumulation steps
    train_loader, test_loader, actual_batch_size, accumulation_steps = prepare_loaders(
        train_loader.dataset, 
        test_loader.dataset, 
        original_batch_size
    )
    
    # Create batch configuration message first
    batch_config_message = (
        f"\n{'='*50}\n"
        f"Batch Configuration for {model_name}:\n"
        f"{'─'*50}\n"
        f"├─ Requested: {original_batch_size}\n"
        f"├─ Actual: {actual_batch_size}\n"
        f"├─ Accumulation steps: {accumulation_steps}\n"
        f"└─ Effective: {actual_batch_size * accumulation_steps}\n"
        f"{'='*50}\n"
    )
    
    # Initialize log messages with batch config
    training_logs[model_name]['log_messages'] = [batch_config_message]
    
    # Update batch size information in progress
    training_progress[model_name].update({
        'batch_size': actual_batch_size,
        'effective_batch_size': actual_batch_size * accumulation_steps
    })
    
    # Wait for start signal
    start_event.wait()
    
    all_epoch_logs = []
    
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        # Use lock when updating shared progress
        with progress_lock:
            training_progress[model_name].update({
                'current_epoch': epoch + 1,
                'progress': 0,
                'current_loss': 0,
                'current_accuracy': 0
            })
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = model(data)
            loss = criterion(output, target) / accumulation_steps
            loss.backward()
            
            # Update weights only after accumulating enough gradients
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            running_loss += loss.item() * accumulation_steps
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Use lock when updating progress
            with progress_lock:
                progress = (batch_idx + 1) / len(train_loader) * 100
                training_progress[model_name].update({
                    'progress': progress,
                    'current_loss': running_loss / (batch_idx + 1),
                    'current_accuracy': 100. * correct / total,
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_model(model, test_loader)
        
        # Store metrics for plotting
        training_logs[model_name]['loss'].append(epoch_loss)
        training_logs[model_name]['accuracy'].append(epoch_acc)
        training_logs[model_name]['test_loss'].append(test_loss)
        training_logs[model_name]['test_accuracy'].append(test_acc)
        
        # Create epoch summary with better formatting
        epoch_log = (
            f"\n{'─'*50}\n"
            f"Epoch {epoch+1}/{n_epochs}\n"
            f"{'─'*50}\n"
            f"├─ Training\n"
            f"│  ├─ Loss: {epoch_loss:.4f}\n"
            f"│  └─ Accuracy: {epoch_acc:.2f}%\n"
            f"│\n"
            f"└─ Testing\n"
            f"   ├─ Loss: {test_loss:.4f}\n"
            f"   └─ Accuracy: {test_acc:.2f}%\n"
            f"{'─'*50}\n"
        )
        all_epoch_logs.append(epoch_log)
        
        # Update complete log history
        training_logs[model_name]['log_messages'] = [batch_config_message] + all_epoch_logs
        
        # Update progress with test metrics
        with progress_lock:
            training_progress[model_name].update({
                'current_test_loss': test_loss,
                'current_test_accuracy': test_acc
            })

def get_optimizer(model, params):
    lr = float(params['learning_rate'])
    if params['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    return optim.SGD(model.parameters(), lr=lr)

@app.route('/get_training_progress')
def get_training_progress():
    print("Training Progress Data:", training_progress)  # Debug print
    return jsonify(training_progress)

# Add this route to get detailed logs
@app.route('/get_training_logs')
def get_training_logs():
    print("Training Logs Data:", {
        'model1': training_logs['model1']['log_messages'][-50:],
        'model2': training_logs['model2']['log_messages'][-50:]
    })  # Debug print
    return jsonify({
        'model1': training_logs['model1']['log_messages'][-50:],
        'model2': training_logs['model2']['log_messages'][-50:]
    })

@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    data = request.get_json()
    dataset_type = data['dataset_type']
    num_images = int(data['num_images'])
    
    # Get dataset
    train_dataset, test_dataset = load_dataset()
    dataset = train_dataset if dataset_type == 'train' else test_dataset
    
    # Randomly select indices
    indices = torch.randperm(len(dataset))[:num_images]
    
    predictions = []
    for idx in indices:
        image, true_label = dataset[idx]
        
        # Get predictions from both models
        with torch.no_grad():
            # Add batch dimension and ensure it's a tensor
            image_tensor = image.unsqueeze(0)
            
            output1 = models['model1'](image_tensor)
            output2 = models['model2'](image_tensor)
            
            prob1 = F.softmax(output1, dim=1)
            prob2 = F.softmax(output2, dim=1)
            
            pred1 = output1.argmax(dim=1).item()
            pred2 = output2.argmax(dim=1).item()
            
            conf1 = prob1[0][pred1].item() * 100
            conf2 = prob2[0][pred2].item() * 100
        
        # Convert image to base64 for display
        img_array = (image.squeeze().numpy() * 255).astype('uint8')
        img_pil = Image.fromarray(img_array)
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Get labels based on dataset type
        label_map = get_label_map(current_dataset)
        
        # Handle true_label whether it's a tensor or int
        true_label_idx = true_label.item() if torch.is_tensor(true_label) else true_label
        
        predictions.append({
            'image': img_str,
            'model1': {
                'label': label_map[pred1],
                'confidence': conf1
            },
            'model2': {
                'label': label_map[pred2],
                'confidence': conf2
            },
            'true_label': label_map[true_label_idx]
        })
    
    return jsonify(predictions)

def get_label_map(dataset_type):
    if dataset_type == 'mnist':
        return {i: str(i) for i in range(10)}
    else:  # Fashion MNIST
        return {
            0: 'T-shirt/Top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot'
        }

if __name__ == '__main__':
    app.run(debug=True) 