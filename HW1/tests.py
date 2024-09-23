import unittest
import numpy as np
import mlp
from numpy.testing import assert_allclose
import pickle as pk

seed = 10417617

with open("tests.pk", "rb") as f: tests = pk.load(f)

TOLERANCE = 1e-5

# to run one test: python -m unittest tests.TestLinearMap
# to run all tests: python -m unittest tests

class TestLinearMap(unittest.TestCase):
  def test(self):
    weights = tests["linmap_weights"]
    bias = tests["linmap_bias"]
    sl = mlp.LinearMap(18, 100, alpha=0)
    sl.loadparams(np.array(weights), np.array(bias))

    test1 = np.arange(18).reshape((18, 1))
    result = tests["linmap_test1"]
    assert_allclose(sl.forward(test1), result, atol=TOLERANCE)

    test2 = np.arange(100).reshape((100, 1))
    result = tests["linmap_test2"]
    assert_allclose(sl.backward(test2), result, atol=TOLERANCE)

    sl.zerograd()

    test3 = np.arange(36).reshape((18, 2))
    result = tests["linmap_test3"]
    assert_allclose(sl.forward(test3), result, atol=TOLERANCE)

    test4 = np.arange(200).reshape((100, 2))
    result = tests["linmap_test4"]
    assert_allclose(sl.backward(test4), result, atol=TOLERANCE)


class TestMomentum(unittest.TestCase):
  def test(self):
    weights = tests["mom_weights"]
    bias = tests["mom_bias"]
    sl = mlp.LinearMap(18, 100, alpha=0.5, lr=0.01)
    sl.loadparams(np.array(weights), np.array(bias))
    test5 = np.arange(36).reshape((18, 2))
    
    result = tests["mom_test5"]
    assert_allclose(sl.forward(test5), result ,atol=TOLERANCE)
    
    test6 = np.arange(200).reshape((100, 2))
    sl.backward(test6)
    sl.step()
    sl.zerograd()
    sl.backward(test6)
    sl.step()
    result_weight = tests["mom_test6a"]
    result_bias = tests["mom_test6b"]
    assert_allclose(sl.getW(), result_weight, atol=TOLERANCE)
    assert_allclose(sl.getb(), result_bias, atol=TOLERANCE)

    
class TestLeakyReLU(unittest.TestCase):
  def test(self):
    sl = mlp.LeakyReLU(alpha=0.1, dropout_probability=0)
    
    test7 = (np.arange(36).reshape((18, 2))-18).astype('float64')
    assert_allclose(sl.forward(test7, train=False), tests["leakyrelu_test7a"])
    assert_allclose(sl.forward(test7), tests["leakyrelu_test7b"])
    
    test8 = (np.arange(36).reshape((18, 2))-22).astype('float64')
    assert_allclose(sl.backward(test8), tests["leakyrelu_test8"])


class TestLeakyReLUDropout(unittest.TestCase):
  def test(self):
    sl = mlp.LeakyReLU(alpha=0.1, dropout_probability=0.5)

    test9 = (np.arange(36).reshape((18, 2))-18).astype('float64')
    assert_allclose(sl.forward(test9, train=False), tests["leakyreludropout_test9a"])
    np.random.seed(seed)
    assert_allclose(sl.forward(test9), tests["leakyreludropout_test9b"])
    
    test10 = (np.arange(36).reshape((18, 2))-22).astype('float64')
    assert_allclose(sl.backward(test10), tests["leakyreludropout_test10"])
    

class TestLoss(unittest.TestCase):
  def test(self):
    sl = mlp.SoftmaxCrossEntropyLoss()

    np.random.seed(1)
    logits = np.random.uniform(-1, 1, [18, 2])
    labels = np.zeros(logits.shape)
    labels[3, 0], labels[15, 1] = 1, 1

    logits = np.arange(36).reshape((18, 2)).astype('float64')
    logits[:, 0] = logits[:, 0] / np.sum(logits[:, 0])
    logits[:, 1] = logits[:, 1] / np.sum(logits[:, 1])

    labels = np.zeros((18, 2)).astype('float64')
    labels[0][0] = 1
    labels[1][1] = 1

    result = tests["softmaxce_test11a"]
    assert_allclose(sl.forward(logits, labels), result, atol=TOLERANCE)
    result = tests["softmaxce_test11b"]
    assert_allclose(sl.backward(), result, atol=TOLERANCE)


class TestSingleLayerMLP(unittest.TestCase):
  def test(self):
    data = [np.arange(20).reshape((20, 1)), np.arange(20).reshape((20, 1))-1]
    np.random.seed(seed)
    ann = mlp.SingleLayerMLP(20, 19,
                             hiddenlayer=100, alpha=0.1, dropout_probability=0.5)

    Ws = tests["sl_initweights"]
    bs = tests["sl_initbias"]
    ann.loadparams(Ws, bs)

    np.random.seed(seed)
    ann.forward(data[0])
    ann.backward(np.arange(19).reshape((19, 1)))
    ann.step()

    result = tests["sl_w0"]
    for w, result_w in zip(ann.getWs(), result):
      assert_allclose(w, result_w, atol=TOLERANCE)
    result = tests["sl_b0"]
    for b, result_b in zip(ann.getbs(), result):
      assert_allclose(b, result_b, atol=TOLERANCE)

    ann.zerograd()
    ann.backward(np.arange(19).reshape((19, 1))+1)
    ann.step()

    result = tests["sl_w1"]
    for w, result_w in zip(ann.getWs(), result):
      assert_allclose(w, result_w, atol=TOLERANCE)
    result = tests["sl_b1"]
    for b, result_b in zip(ann.getbs(), result):
      assert_allclose(b, result_b, atol=TOLERANCE)


class TestTwoLayerMLP(unittest.TestCase):
  def test(self):
    data = [np.arange(20).reshape((20, 1)), np.arange(20).reshape((20, 1))-1]
    np.random.seed(seed)
    ann = mlp.TwoLayerMLP(20, 19,
                          hiddenlayers=[100, 100], alpha=0.1, dropout_probability=0.5)
    Ws = tests["tl_initweights"]
    bs = tests["tl_initbias"]
    ann.loadparams(Ws, bs)

    np.random.seed(seed)
    ann.forward(data[0])
    ann.backward(np.arange(19).reshape((19, 1)))
    ann.step()

    result = tests["tl_w0"]
    for w, result_w in zip(ann.getWs(), result):
      assert_allclose(w, result_w, atol=TOLERANCE)
    result = tests["tl_b0"]
    for b, result_b in zip(ann.getbs(), result):
      assert_allclose(b, result_b, atol=TOLERANCE)

    ann.zerograd()
    ann.backward(np.arange(19).reshape((19, 1))+1)
    ann.step()

    result = tests["tl_w1"]
    for w, result_w in zip(ann.getWs(), result):
      assert_allclose(w, result_w, atol=TOLERANCE)
    result = tests["tl_b1"]
    for b, result_b in zip(ann.getbs(), result):
      assert_allclose(b, result_b, atol=TOLERANCE)