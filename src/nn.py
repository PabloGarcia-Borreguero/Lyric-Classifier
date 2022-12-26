import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall
from sklearn.preprocessing import normalize, StandardScaler
import matplotlib.pyplot as plt





class NeuralNetwork(nn.Module):

    def __init__(
        self, 
        input_size: int = 103, 
        output_size: int = None,
    )-> None:
        super(NeuralNetwork, self).__init__()
        self.seq = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Flatten(1, -1),
            nn.Softmax(dim=1)
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dim = output_size

    def forward(self, x):

        x = x.to(torch.float32)

        return self.seq(x)

    def begin_training(self, X_train: np.ndarray, y_train, X_test, y_test):

        model = self.to(self.device)

        X_train = self.scale_and_standardize(X_train)

        X_test = self.scale_and_standardize(X_test)

        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
        model.train()
        batch_size = 200
        n_iters = 4000
        num_epochs = n_iters / (len(X_train) / batch_size)
        num_epochs = int(num_epochs)

        criterion = nn.BCELoss()


        training_data = [(input,label) for input, label in zip(X_train, y_train)]

        testing_data = [(input,label) for input, label in zip(X_test, y_test)]

        train_loader = torch.utils.data.DataLoader(dataset=training_data, 
                                           batch_size=batch_size, 
                                           shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=testing_data, 
                                          batch_size=batch_size, 
                                          shuffle=False)
        iter = 0
        loss_values = []
        for epoch in range(num_epochs):
            for i, (embeddings, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                embeddings = embeddings.to(torch.float32)
                labels = labels.to(torch.float32)
                # Forward pass
                outputs = model(embeddings)
                # Compute Loss
                loss = criterion(outputs, labels)
            
                loss_values.append(loss)
                # Backward pass
                loss.backward()
                # Update parameters
                optimizer.step()

                iter += 1

                if iter %  500 == 0:
                    # Calculate Accuracy         
                    correct = 0
                    total = 0
                
                    # Iterate through test dataset
                    for embeddings, labels in test_loader:
                        # Load images with gradient accumulation capabilities
                        #images = images.view(-1, 28*28).requires_grad_()

                        # Forward pass only to get logits/output
                        outputs = model(embeddings)

                        thresh = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                        precision = []
                        recall = []
                        for t in thresh:
                            o = outputs.clone().detach()
                            # Get predictions from the maximum value
                            # Total number of labels
                            o[o >= t] = 1
                            o[o< t] = 0

                            # Total number of labels
                            total += labels.nelement()

                            # Total correct predictions
                            correct += (o == labels).sum()

                            prec = MultilabelPrecision(
                                num_labels=self.output_dim
                            )
                            precision.append(prec(o, labels))
                            rec = MultilabelRecall(
                                num_labels=self.output_dim
                            )
                            recall.append(rec(o, labels))
                    
                    #recall_max = max(recall)
                    #precision_max = max(precision)
                    accuracy = correct / total

                    # Print Loss
                    print('Iteration: {}. Loss: {}. Accuracy: {}.'.format(iter, loss.item(), accuracy, ))

        torch.save(model, "D:/TM_proyect/assets/model.pt")

    
    def evaluate(self, model, X_test, y_test):
        model.eval()
        y_pred = model(X_test)
        after_train = self.criterion(y_pred.squeeze(), y_test) 
        print('Test loss after Training' , after_train.item())

    
    def scale_and_standardize(self, X):

        sc_X = StandardScaler()
        sc_X = sc_X.fit_transform(X)

        X =  normalize(sc_X, axis=1)

        return X

    def predict(self, embeddings):
        self.eval()
        # model is self(VGG class's object)
        
        count = embeddings.shape[0]
        result_np = []
            
        for embedding in embeddings:
            output = self(embedding)
        
        return result_np

    

    
        
