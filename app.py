import torch
import cloudpickle

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = cloudpickle.load(open('model.pkl', 'rb'))

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: list) -> dict:
    global model
    
    # Run the model
    result = model(torch.tensor(model_inputs))

    # Return the results as a list
    return result.detach().numpy().tolist()
