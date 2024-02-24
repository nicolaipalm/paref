import gpytorch
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


class ExactGP0(gpytorch.models.ExactGP):
    """
    # TBA: add
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood):
        super(ExactGP0, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class Gpr0Torch:
    def __init__(self, training_iter=1000, learning_rate=0.1):
        self._training_iter = training_iter
        self._learning_rate = learning_rate
        # with no white noise
        # self._likelihood = gpytorch.likelihoods.GaussianLikelihood(
        #    noise_constraint=gpytorch.constraints.Interval(
        #       lower_bound=0.00000001, upper_bound=0.001))
        # with learned white noise
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._model = None
        self.hyperparameters = None

    def predict_torch(
            self, pred_x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Doing predictions based on the models and the prediction data.
        Input and output need to be tensors.
        Note that the output need to be of dimension one.
        """
        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # make predictions for pred_x
            observed_pred = self._likelihood(self._model(pred_x))

        return observed_pred

    def train_torch(self, train_x: torch.Tensor, train_y: torch.Tensor) -> bool:
        """
        Train the model with torch tensors as input and output.
        Note that the output need to be of dimension one.
        """
        # Initialize likelihood (with no constraints) and _model
        model = ExactGP0(train_x, train_y, self._likelihood)

        # Set on the training mode of the models
        model.train()
        self._likelihood.train()

        # Use the adam optimizer of torch
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, model)

        # store hyperparameters of the training
        hyper_parameter = {name: [] for name, _ in model.named_parameters()}
        hyper_parameter['loss'] = []

        # Start the training
        for _ in tqdm(range(self._training_iter)):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from _model
            output = model(train_x)
            # Calc loss and backpropagation gradients
            loss = -mll(output, train_y)
            loss.backward()

            optimizer.step()
            for name, parameter in model.named_parameters():
                if model.constraint_for_parameter_name(name) is not None:
                    hyper_parameter[name].append(
                        model.constraint_for_parameter_name(name)
                        .transform(parameter)
                        .item()
                    )

                else:
                    hyper_parameter[name].append(parameter.item())

            # Appending loss
            hyper_parameter['loss'].append(loss.item())

        self.hyperparameters = hyper_parameter  # store hyperparameters
        self._model = model
        return True

    def _train_with_attention(
            self, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> dict:
        """
        This method is used to train _and_ track the progress of the hyperparameters,
        i.e. it stores the hyperparameters in each step and prints the values in each step.
        You can use this method to check if the training was succesfull or to debug your training.
        See example>training_with_attention for an example to visualize this progress.
        """
        # Initialize likelihood (with no constraints) and _model
        model = ExactGP0(train_x, train_y, self._likelihood)

        # Set on the training mode of the models
        model.train()
        self._likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, model)

        hyper_parameter = {name: [] for name, _ in model.named_parameters()}
        hyper_parameter['loss'] = []

        # Start the training
        for _ in tqdm(range(self._training_iter)):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from _model
            output = model(train_x)
            # Calculate loss and backpropagation gradients
            loss = -mll(output, train_y)
            loss.backward()

            optimizer.step()
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' %
            #      (i + 1, self._training_iter, loss.item(),
            #       model.covar_module.base_kernel.lengthscale.item(),
            #       model.likelihood.noise.item()))
            # TBA_later: delete raw in hyperparameters since it is transformed
            # Appending hyperparameters
            for name, parameter in model.named_parameters():
                if model.constraint_for_parameter_name(name) is not None:
                    hyper_parameter[name].append(
                        model.constraint_for_parameter_name(name)
                        .transform(parameter)
                        .item()
                    )

                else:
                    hyper_parameter[name].append(parameter.item())

            # Appending loss
            hyper_parameter['loss'].append(loss.item())

        self._model = model
        return hyper_parameter

    def save_state(self, state_name) -> bool:
        """
        Save the trained hyperparameter to a .pth file.
        """
        torch.save(self._model.state_dict(), state_name + '.pth')
        return True

    def load_state(
            self, state_path: str, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> bool:
        """
        Load a set of hyperparameters together with the training set(!) into the models.
        """
        state_dict = torch.load(state_path)
        model = ExactGP0(train_x, train_y, self._likelihood)
        model.load_state_dict(state_dict)
        self._model = model
        return True


class GPR:
    def __init__(self,
                 training_iter: int = 1000,
                 learning_rate=0.05, ):
        self._models = None
        self._training_iter = training_iter
        self._learning_rate = learning_rate
        self._means = None

    def train(self, train_x: np.ndarray, train_y: np.ndarray):
        # prepossessing
        self._means = np.mean(train_y, axis=0)
        self._std = np.std(train_y, axis=0)
        # training
        train_x = torch.Tensor(train_x)
        models = []
        for output in (train_y.T - self._means.reshape(-1, 1)) / self._std.reshape(-1, 1):
            model = Gpr0Torch(
                training_iter=self._training_iter, learning_rate=self._learning_rate
            )
            model.train_torch(train_x, torch.Tensor(output))
            models.append(model)
        self._models = models

        # Check if training converged
        losses = np.array([hp['loss'] for hp in self.info])
        losses_last = np.array([hp['loss'][int(self._training_iter * 0.5):] for hp in self.info])

        self._model_convergence = (np.max(losses_last, axis=1) - np.min(losses_last, axis=1)) / (
                np.max(losses, axis=1) - np.min(losses, axis=1)) * 2
        return True

    @property
    def model_convergence(self):
        return self._model_convergence

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = torch.Tensor([x.tolist()])
        return np.array(
            [model.predict_torch(x).mean.numpy()[0] for model in self._models]
        ).T * self._std + self._means

    def std(self, x: np.ndarray) -> np.ndarray:
        x = torch.Tensor([x.tolist()])
        return np.array(
            [model.predict_torch(x).stddev.numpy()[0] for model in self._models]
        ).T

    @property
    def info(self):
        return [gpr.hyperparameters for gpr in self._models]

    def plot_loss(self):
        fig, axs = plt.subplots(1, len(self._models))
        fig.suptitle('Loss of GPR model(s)')
        if len(self._models) > 1:
            for i in range(len(self._models)):
                axs[i].plot(self._models[i].hyperparameters['loss'])
                axs[i].set_title(f'GPR model {i}')

            axs[0].set(xlabel='Training iteration', ylabel='loss')
        else:
            axs.plot(self._models[0].hyperparameters['loss'])
            axs.set(xlabel='Training iteration', ylabel='loss')
        plt.show()
