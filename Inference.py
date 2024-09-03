import streamlit as st
import torch

# Define the model class
class CpGPredictor(torch.nn.Module):
    def __init__(self):
        super(CpGPredictor, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=5, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYER, batch_first=True)
        self.classifier = torch.nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out[:, -1, :])
        return logits

# Initialize the model
model = CpGPredictor()

# Load the model state dictionary
model.load_state_dict(torch.load('model.pth'))

# Set the model to evaluation mode
model.eval()

# Streamlit app
st.title('CpG Detector')

input_sequence = st.text_input('Enter DNA Sequence (N, A, C, G, T):')

if input_sequence:
    # Convert input sequence to tensor
    int_seq = list(dnaseq_to_intseq(input_sequence))
    input_tensor = torch.tensor([int_seq], dtype=torch.long)
    input_tensor = torch.nn.functional.one_hot(input_tensor, num_classes=5).float()
    
    # Predict
    with torch.no_grad():
        prediction = model(input_tensor)
    
    st.write(f'Predicted Number of CpGs: {prediction.item()}')