import numpy as np

### Main functions in Step 2 of pdf used in the project. 

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: final loss value after max_iters of GD
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = mean_squared_error_g(y, tx, w)
        w = w - gamma * grad
    loss = mean_squared_error_loss(y, tx, w)

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: final loss value after max_iters of SGD
    """
    w = initial_w
    for n_iter in range(max_iters):
        index = np.random.choice(y.shape[0], 1, replace=False) # batch_size=1
        y_B = y[index]
        tx_B = tx[index, :]
        stoch_grad = mean_squared_error_sg(y_B, tx_B, w)
        w = w - gamma * stoch_grad
    loss = mean_squared_error_loss(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mse scalar.
    """
    # normal equations: (X^T X) w = X^T y
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)

    e = y - tx.dot(w)
    loss = 0.5 * np.mean(e**2)
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mse scalar.
    """
    N, D = tx.shape

    # normal equations: (X^T X + lambda) w = X^T y
    a = tx.T.dot(tx) + 2 * N * lambda_ * np.eye(D)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    e = y - tx.dot(w)
    loss = 0.5 * np.mean(e**2)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm for logistic regression.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: final loss value after max_iters of GD
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = NLL_g(y, tx, w)
        w = w - gamma * grad
    loss = NLL_loss(y, tx, w)

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm for regularised logistic regression.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: final loss value after max_iters of GD
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = NLL_g(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = NLL_loss(y, tx, w)

    return w, loss



### Auxiliary functions used in the main functions above.

def mean_squared_error_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        loss: the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    loss = 0.5 * np.mean(e ** 2)
    return loss

def mean_squared_error_g(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = len(y)
    e = y - tx.dot(w)
    return (-1/N) * tx.T.dot(e)

def mean_squared_error_sg(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    B = len(y)
    e = y - tx.dot(w)
    return (-1/B) * tx.T.dot(e)

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1 / (1 + np.exp(-t))


def NLL_loss(y, tx, w):
    """compute the cost by negative log likelihood for y in {0, 1}. 

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a non-negative loss

    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    z = tx @ w               
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return float(loss)

def NLL_g(y, tx, w):
    """compute the gradient of negative log likelihood loss for y in {0, 1}.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a vector of shape (D, )
    """
    N = y.shape[0]

    return 1/N * tx.T @ (sigmoid(tx @ w) - y)


