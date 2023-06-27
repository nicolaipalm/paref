import gpytorch
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


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

        # Start the training
        for i in tqdm(range(self._training_iter)):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from _model
            output = model(train_x)
            # Calc loss and backpropagation gradients
            loss = -mll(output, train_y)
            loss.backward()

            optimizer.step()
        self._model = model
        return True

    def _train_with_attention(
        self, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> dict:
        # Initialize likelihood (with no constraints) and _model
        """
        This method is used to train _and_ track the progress of the hyperparameters,
        i.e. it stores the hyperparameters in each step and prints the values in each step.
        You can use this method to check if the training was succesfull or to debug your training.
        See example>training_with_attention for an example to visualize this progress.
        """
        model = ExactGP0(train_x, train_y, self._likelihood)

        # Set on the training mode of the models
        model.train()
        self._likelihood.train()

        # Use the adam optimizer of torch
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, model)

        hyper_parameter = {name: [] for name, _ in model.named_parameters()}
        hyper_parameter['loss'] = []

        # Start the training
        for i in tqdm(range(self._training_iter)):
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
            # TBA_later: delete raw in hyper parameters since it is transformed
            # Appending hyper parameter
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
    def __init__(self, training_iter: int = 1000, learning_rate=0.1):
        self._models = None
        self._training_iter = training_iter
        self._learning_rate = learning_rate

    def train(self, train_x: np.ndarray, train_y: np.ndarray):
        train_x = torch.Tensor(train_x)
        models = []
        for output in train_y.T:
            train_y = torch.Tensor(output)
            model = Gpr0Torch(
                training_iter=self._training_iter, learning_rate=self._learning_rate
            )
            model.train_torch(train_x, train_y)
            models.append(model)
        self._models = models
        return True

    def train_with_attention(self, train_x: np.ndarray, train_y: np.ndarray):
        train_x = torch.Tensor(train_x)
        models = []
        for output in train_y.T:
            train_y = torch.Tensor(output)
            model = Gpr0Torch(
                training_iter=self._training_iter, learning_rate=self._learning_rate
            )
            hyper_parameters = model._train_with_attention(train_x, train_y)
            ####
            f, axs = plt.subplots(
                2, int(len(hyper_parameters) / 2 + 1), figsize=(12, 8)
            )
            f.suptitle('Hyper parameters')
            ticker_x = 0
            ticker_y = 0
            print(hyper_parameters.keys())
            for hyper_parameter in hyper_parameters.keys():
                axs[ticker_x, ticker_y].plot(hyper_parameters.get(hyper_parameter))
                axs[ticker_x, ticker_y].set(
                    xlabel='training_iter', ylabel=hyper_parameter
                )
                ticker_x += 1
                ticker_x = ticker_x % 2
                if ticker_x == 0:
                    ticker_y += 1
                    ticker_x = 0
            plt.show()
            ####

            # model.train_torch(train_x, train_y)
            models.append(model)

        self._models = models

        return True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = torch.Tensor([x.tolist()])
        return np.array(
            [model.predict_torch(x).mean.numpy()[0] for model in self._models]
        ).T

    def std(self, x: np.ndarray) -> np.ndarray:
        x = torch.Tensor([x.tolist()])
        return np.array(
            [model.predict_torch(x).stddev.numpy()[0] for model in self._models]
        ).T
