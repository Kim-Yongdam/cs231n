import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train) :
    score_i = X[i].dot(W)
    adjust_score = score_i - max(score_i)
    loss_i = - adjust_score[y[i]] + np.log(sum(np.exp(adjust_score)))
    loss += loss_i

    for j in range(num_classes) :
      softmax_loss = np.exp(adjust_score[j]) / sum(np.exp(adjust_score))

      if j == y[i] :
        dW[:, j] += (-1 + softmax_loss) * X[i]

      else :
        dW[:, j] += softmax_loss * X[i]


  loss = loss / float(num_train) + 0.5 * reg * np.sum(W * W)
  dW = dW / float(num_train) + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]

  score_i = X.dot(W)
  adjust_score = score_i - np.max(score_i, axis=1, keepdims = True)
  softmax_loss = np.exp(adjust_score) / np.sum(np.exp(adjust_score), axis=1, keepdims = True)

  loss = np.sum(-np.log(softmax_loss[np.arange(num_train), y]))
  loss = loss / float(num_train) + 0.5 * reg * np.sum(W * W)

  dS = np.zeros_like(softmax_loss)
  dS[np.arange(num_train), y] = 1

  dW = (X.T).dot(softmax_loss - dS)
  dW = dW / float(num_train) + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

