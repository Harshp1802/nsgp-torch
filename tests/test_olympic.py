import pytest
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def test_model():
    from nsgp import NSGP
    from nsgp.utils.inducing_functions import f_kmeans
    import torch
    import numpy as np
    np.random.seed(7)
    torch.manual_seed(7)
    num_low = 25
    num_high = 25
    gap = -.1
    noise = 0.0001
    # X = torch.vstack((torch.linspace(-1, -gap/2.0, num_low)[:, np.newaxis],
    #                   torch.linspace(gap/2.0, 1, num_high)[:, np.newaxis])).reshape(-1, 1)
    # y = torch.vstack((torch.zeros((num_low, 1)), torch.ones((num_high, 1))))
    # scale = torch.sqrt(y.var())
    # offset = y.mean()
    # y = ((y-offset)/scale).reshape(-1, 1)

    # X_new = torch.linspace(-1, 1, 100).reshape(-1, 1)
    scalerX, scalery = StandardScaler(), StandardScaler()
    data = pd.read_csv("../olympic.csv", index_col=0)
    data = data.sample(frac=1)
    X = scalerX.fit_transform(data["X"].to_numpy().reshape(-1, 1))
    y = scalery.fit_transform(data["y"].to_numpy().reshape(-1, 1))

    # X = torch.Tensor(data['X'].to_numpy()).reshape(-1, 1)
    # y = torch.Tensor(data['y'].to_numpy()).flatten()
    X = torch.Tensor(X).reshape(-1, 1)
    y = torch.Tensor(y).flatten()
    scale = torch.sqrt(y.var())
    offset = y.mean()
    y = ((y-offset)/scale).reshape(-1, 1)

    # X_new = torch.linspace(-1, 1, 100).reshape(-1, 1)
    X_new = np.linspace(1870, 2030, 200).reshape(-1, 1)
    X_new = torch.Tensor(scalerX.transform(X_new))
    X_bar = f_kmeans(X, num_inducing_points=5, random_state=7)
    model = NSGP(X, y, X_bar=X_bar, jitter=10**-5)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    # optim = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []
    model.train()
    for _ in range(180):
        optim.zero_grad()
        loss = model.nlml()
        losses.append(loss.item())
        loss.backward()
        optim.step()

    print(losses)
    model.eval()
    with torch.no_grad():
        y_new, y_var = model.predict(X_new)
        y_std2 = y_var.diagonal()**0.5 * 2

        X = scalerX.inverse_transform(X)
        X_new = scalerX.inverse_transform(X_new)
        y_new = scalery.inverse_transform(y_new)
        # y_std2 = scalery.inverse_transform(y_std2)
        y = scalery.inverse_transform(y)

        fig, ax = plt.subplots(3, 1, figsize=(10, 16))
        ax[0].scatter(X, y)
        # ax[0].plot(X_new.numpy(), y_new.numpy())
        ax[0].plot(X_new, y_new)
        ax[0].fill_between(X_new.ravel(), y_new.ravel() -
                           y_std2.numpy(), y_new.ravel()+y_std2.numpy(), alpha=0.5)

        ax[2].plot(losses)

        ax[1].plot(X_new, model.get_LS(
            torch.Tensor(scalerX.transform(X_new)))[0])

        fig.savefig('./test_olympic.pdf')
