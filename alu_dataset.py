import torch
from torch.utils.data import Dataset

class Dataset_alu(Dataset):
    
    def __init__(self, data, delta_time):
        self.timesteps = 1
        self.DT = delta_time

        self.data = data
        
        self.x, self.x_mean, self.x_std = self.organize_features(data)
        
        self.y, self.y_mean, self.y_std, self.y_non_norm = self.organize_target(data)        
        
        self.x_non_norm = torch.flatten(data,start_dim=0, end_dim=1)
        
        self.n_samples = self.x.shape[0]
        
        
    def __getitem__(self, index):
        
        X = self.x[index]
        
        Y = self.y[index]
        
        return X, Y
    
    def __len__(self):
        return self.n_samples
    
    def organize_features(self, data):
        #standard normalize input: (x-mean)/std(x)
        temp = torch.flatten(data, start_dim=0, end_dim=1)
        mean = torch.mean(temp, axis=0)
        std = torch.std(temp, axis=0)

        t_steps = data.shape[1]
        no_sim = data.shape[0]

        temp_input = torch.empty(no_sim, t_steps-1, data.shape[2])

        for i in range(data.shape[0]):
            temp_input[i,:,:] = (data[i,0:t_steps-1,:] - mean)/std

        x = torch.flatten(temp_input, start_dim=0, end_dim = 1)

        return x, mean, std


    def organize_target(self, data):
        #Standard normalize output: (y- mean(y))/std(y)

        t_steps = data.shape[1]
        no_sim = data.shape[0]

        temp_target = torch.empty(data.shape[0], data.shape[1] - 1, 8)

        for i in range(no_sim):
            for j in range(t_steps-1):
                temp_target[i,j,:] = (data[i,j+1,0:8] - data[i,j,0:8])/self.DT

        y_not_norm = torch.flatten(temp_target,start_dim=0, end_dim=1)

        mean = torch.mean(y_not_norm, axis=0)
        std = torch.std(y_not_norm, axis=0)
        y = (y_not_norm - mean)/std

        return y, mean, std, y_not_norm


class residual_dataset_alu(Dataset):
    def __init__(self, X_true, X_pbm_dot, delta_time,feature_normalization='std_norm', output_normalization ='std_norm'):
        # Handle small values - should not be normalized
        self.DT = delta_time

        self.xdot_true, self.xdot_mean, self.xdot_std, self.xdot_not_norm, self.xdot_min, self.xdot_max = self.organize_dxdt_true(X_true)


        if feature_normalization=='std_norm':
            self.x, self.x_mean, self.x_std = self.organize_features_std_norm(X_true)

        elif feature_normalization=='min_max':
            self.x, self.x_min, self.x_max = self.organize_features_min_max(X_true)

        if output_normalization=='std_norm':
            self.y, self.y_not_norm = self.organize_output_std_norm(X_pbm_dot)
        elif output_normalization=='min_max':
            self.y, self.y_not_norm = self.organize_output_min_max(X_pbm_dot)
        self.n_samples = self.x.shape[0]

    def __getitem__(self,index):
        
        X = self.x[index]
        
        Y = self.y[index]
        
        return X, Y
    
    def __len__(self):
        return self.n_samples


    def organize_dxdt_true(self, X_true):
        #Standard normalize output: (y- mean(y))/std(y)

        t_steps = X_true.shape[1]
        no_sim = X_true.shape[0]

        temp_target = torch.empty(X_true.shape[0], X_true.shape[1] - 1, 8)

        for i in range(no_sim):
            for j in range(t_steps-1):
                temp_target[i,j,:] = (X_true[i,j+1,0:8] - X_true[i,j,0:8])/self.DT

        xdot_not_norm = torch.flatten(temp_target,start_dim=0, end_dim=1)

        #For std_norm
        mean = torch.mean(xdot_not_norm, axis=0)
        std = torch.std(xdot_not_norm, axis=0)

        #For min_max
        min = torch.min(xdot_not_norm,dim=0)[0]
        max = torch.max(xdot_not_norm,dim=0)[0]
        xdot = (xdot_not_norm - mean)/std

        return xdot, mean, std, xdot_not_norm, min, max
    
    def organize_features_std_norm(self, data):
        #standard normalize input: (x-mean)/std(x)
        temp = torch.flatten(data, start_dim=0, end_dim=1)
        mean = torch.mean(temp, axis=0)
        std = torch.std(temp, axis=0)

        t_steps = data.shape[1]
        no_sim = data.shape[0]

        temp_input = torch.empty(no_sim, t_steps-1, data.shape[2])

        for i in range(data.shape[0]):
            temp_input[i,:,:] = (data[i,0:t_steps-1,:] - mean)/std

        x = torch.flatten(temp_input, start_dim=0, end_dim = 1)

        return x, mean, std
    
    def organize_features_min_max(self, data):
        temp = torch.flatten(data, start_dim=0,end_dim=1)
        min_x = torch.min(temp,dim=0)[0]
        max_x = torch.max(temp, dim=0)[0]

        t_steps = data.shape[1]
        no_sim = data.shape[0]

        temp_input = torch.empty(no_sim, t_steps-1, data.shape[2])

        for i in range(data.shape[0]):
            temp_input[i,:,:] = (data[i,0:t_steps-1,:] - min_x)/(max_x - min_x)

        x = torch.flatten(temp_input, start_dim=0, end_dim = 1)
        return x, min_x, max_x


    def organize_output_std_norm(self,X_pbm_dot):
        X_pbm_dot_flat = torch.flatten(X_pbm_dot, start_dim=0, end_dim=1)

        residual_not_norm = self.xdot_not_norm - X_pbm_dot_flat
        
        residual = residual_not_norm/self.xdot_std

        return residual, residual_not_norm

    def organize_output_min_max(self,X_pbm_dot):
        X_pbm_dot_flat = torch.flatten(X_pbm_dot, start_dim=0, end_dim=1)

        residual_not_norm = self.xdot_not_norm - X_pbm_dot_flat
        
        residual = residual_not_norm/(self.xdot_max - self.xdot_min)

        return residual, residual_not_norm