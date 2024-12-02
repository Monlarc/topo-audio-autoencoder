import torch
from torch.utils.data import DataLoader
from torch import Generator
from torch.optim import Adam
from tqdm import tqdm 
import os
from glob import glob
from itertools import product
from nsyth_dataset import NSynthDataset
from loss import AutoencoderLoss
import torchaudio
import pickle


def set_seeds(seed):
    torch.manual_seed(seed)
    g = Generator(device='cpu')
    g.manual_seed(seed)
    return g

class Trainer:
    def __init__(self, model, 
                 train_dataset: NSynthDataset, 
                 dev_dataset: NSynthDataset, 
                 test_dataset: NSynthDataset,
                 checkpoint_dir, 
                 best_perf_dict,
                 hypers,
                 seed=511990, 
                 ignore_index=384):
        
        self.g = set_seeds(seed)
        self.model = model
        self.train_loader = DataLoader(train_dataset, shuffle=True, generator=self.g)
        self.dev_loader = DataLoader(dev_dataset, shuffle=True, generator=self.g)
        self.test_loader = DataLoader(test_dataset, shuffle=True, generator=self.g)
        self.best_perf_dict = best_perf_dict
        self.parent_checkpoint_dir = checkpoint_dir
        self.ignore_index = ignore_index
        self.tuned = False
        self.hypers = hypers
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        # self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.loss_fn = AutoencoderLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint_dir):
        list_of_files = glob(f'{checkpoint_dir}/ep*.pth')
        latest =  max(list_of_files, key=os.path.getctime, default=None) if list_of_files else None
        start_epoch = 1
        if latest:
            print(f"Found checkpoint {latest}. Loading...")
            checkpoint = torch.load(latest)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.best_perf_dict = checkpoint['performance']
        else:
            print(f"No checkpoint found. Starting training from scratch.")
        return start_epoch

    def evaluate_model_loss(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inp in tqdm(data_loader, desc="Evaluating"):
                inp = inp
                out, loss_component = self.model(inp)
                loss = self.loss_fn(out, inp, loss_component)
                total_loss += loss['spectral_distance'].item()
        return total_loss / len(data_loader)

    def create_checkpoint_state(self, ep, avg_train_loss):
        return {
            'epoch': ep,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': avg_train_loss,
            'performance': self.best_perf_dict
        }

    def train_model(self, params, max_epochs=5):
        lr = params['lr']
        reg = params['reg']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.loss_fn.extra_penalty_factor = reg

        loss_file_path = "./training_losses.txt"
        with open(loss_file_path, 'w') as file:
            file.write("Epoch,Iteration,Loss\n")
        # self.model.to(self.device)

        checkpoint_dir = os.path.join(self.parent_checkpoint_dir, f'{int(lr*10e4)}_f{int(reg*10e4)}')
        os.makedirs(checkpoint_dir, exist_ok=True)

        start_epoch = self.load_checkpoint(checkpoint_dir)

        count = 0
        frequency = 100
        mini_losses = 0
        for ep in range(start_epoch, max_epochs + 1):
            self.model.train()
            total_loss = 0

            best_audio = []
            worst_audio = []
            os.makedirs(checkpoint_dir + f"./ep{ep}./best", exist_ok=True)
            os.makedirs(checkpoint_dir + f"./ep{ep}./worst", exist_ok=True)
            for inp in tqdm(self.train_loader, desc=f"Epoch {ep}"):
                try:
                    self.optimizer.zero_grad()
                    out, loss_component = self.model(inp)
                    out = out.to(torch.device('cpu'))
                    loss_component = loss_component.to(torch.device('cpu'))
                    loss = self.loss_fn(inp, out, loss_component)
                    loss['spectral_distance'].backward()
                    self.optimizer.step()
                    loss_value = loss['spectral_distance'].item()

                    if loss_value < self.min_loss:
                        best_audio.append((loss_value, inp, out, self.model.get_complex()))
                        self.min_loss = loss_value

                    elif loss_value > self.max_loss:
                        worst_audio.append((loss_value, inp, out, self.model.get_complex()))
                        self.max_loss = loss_value
                        
                    total_loss += loss_value

                    mini_losses += loss_value
                    if count % frequency == 0:
                        print(f"Average loss for the past 100 iterations is: {mini_losses/100}")
                        mini_losses = 0
                    count += 1
                    
                    with open(loss_file_path, 'a') as file:
                        file.write(f"{ep},{count},{loss_value}\n")

                except RuntimeError as e:
                    print("RuntimeError:", e)

            sorted_best = sorted(best_audio, key=lambda x: x[0])
            sorted_worst = sorted(worst_audio, key=lambda x: x[0])

            sample_rate = 16000
            for i, best in enumerate(sorted_best):
                best_path = "../best"
                ranked_best = os.path.join(best_path, f'./{0}')
                os.makedirs(ranked_best, exist_ok=True)
                torchaudio.save(f"{ranked_best}/input.wav", best[1].squeeze(0), sample_rate)
                torchaudio.save(f"{ranked_best}/output.wav", best[2].squeeze(0).detach(), sample_rate)
                self.save_pickle(best[3], f"{ranked_best}/complex.pkl")

            for i, worst in enumerate(sorted_worst):
                worst_path = "../worst"
                ranked_worst = os.path.join(worst_path, f'./{0}')
                os.makedirs(ranked_worst, exist_ok=True)
                torchaudio.save(f"{ranked_worst}/input.wav", worst[1].squeeze(0), sample_rate)
                torchaudio.save(f"{ranked_worst}/output.wav", worst[2].squeeze(0).detach(), sample_rate)
                self.save_pickle(worst_path[3], f"{ranked_worst}/complex.pkl")

            avg_train_loss = total_loss / len(self.train_loader)
            print(f"Average training batch loss: {avg_train_loss}")

            avg_dev_loss = self.evaluate_model_loss(self.dev_loader)
            print(f"Average Dev Loss: {avg_dev_loss}")

            state = self.create_checkpoint_state(ep, avg_train_loss)

            # Save best model checkpoint
            if avg_dev_loss < self.best_perf_dict.get("metric", float('inf')):
                self.best_perf_dict.update(metric=avg_dev_loss, epoch=ep, lr=lr, reg=reg)
                self.save_checkpoint(state, filename=os.path.join(self.parent_checkpoint_dir, "best_model.pth"))
                
            # Save regular checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"ep{ep}.pth")
            self.save_checkpoint(state, filename=checkpoint_path)

        print(f"Best Dev performance: Loss={self.best_perf_dict['metric']:.3f}\n" + 
              f"at epoch {self.best_perf_dict['epoch']} \n" +
              f"with learning rate {self.best_perf_dict['lr']}\n"+
              f"and extra loss term penalty of {self.best_perf_dict['reg']}")

    def tune_hyperparameters(self, param_grid, max_epochs=5):
        param_combinations = list(product(*param_grid.values()))
        param_names = param_grid.keys()

        best_param_set = None
        best_loss = float('inf')

        for param_set in param_combinations:
            # Create a dictionary of the current parameter set
            params = dict(zip(param_names, param_set))
            print(f"Training with parameters: {params}")

            self.train_model(params, max_epochs)
            avg_dev_loss = self.evaluate_model_loss(self.dev_loader)

            # Update best parameters if current loss is lower
            if avg_dev_loss < best_loss:
                best_loss = avg_dev_loss
                best_param_set = params

            self.model.reset_weights()
            
        self.tuned = True
        # Print out the best set of hyperparameters
        print(f"Best hyperparameters: {best_param_set} with loss: {best_loss}")
    
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def save_pickle(self, obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def evaluate_and_save(self, model, filename='report.txt'):
        model.eval()
        total_loss = 0
        best_audio = []
        worst_audio = []
        results_path = './results'
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
        os.makedirs(results_path + "/best", exist_ok=True)
        os.makedirs(results_path + "/worst", exist_ok=True)

        with torch.no_grad():
            for inp in tqdm(model.test_loader, desc="Evaluating"):
                inp = inp.to(model.device)
                out, loss_component = model(inp)
                loss = model.loss_fn(out, inp, loss_component)
                loss_value = loss['spectral_distance'].item()
                if loss_value < 2*self.min_loss:
                    best_audio.append((loss_value, inp, out, model.get_complex()))

                elif loss_value > self.max_loss:
                    worst_audio.append((loss_value, inp, out, model.get_complex()))
                total_loss += loss_value
        
        sorted_best = sorted(best_audio, key=lambda x: x[0])
        sorted_worst = sorted(worst_audio, key=lambda x: x[0])

        sample_rate = 16000
        for i, best in enumerate(sorted_best):
            ranked_best = os.path.join(best_path, f'/{i}')
            os.makedirs(ranked_best, exist_ok=True)
            torchaudio.save(f"{ranked_best}/input.wav", best[1], sample_rate)
            torchaudio.save(f"{ranked_best}/output.wav", best[2], sample_rate)
            self.save_pickle(best[3], f"{ranked_best}/complex.pkl")

        for i, worst in enumerate(sorted_worst):
            ranked_worst = os.path.join(worst_path, f'/{i}')
            os.makedirs(ranked_worst, exist_ok=True)
            torchaudio.save(f"{ranked_worst}/input.wav", worst[1], sample_rate)
            torchaudio.save(f"{ranked_worst}/output.wav", worst[2], sample_rate)
            self.save_pickle(worst_path[3], f"{ranked_worst}/complex.pkl")

        with open(filename, "w") as f:
            f.write(f"Loss on test data: {loss:.4}\n")