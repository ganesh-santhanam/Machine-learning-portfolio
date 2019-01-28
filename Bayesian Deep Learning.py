import theano.tensor as tt  # pymc devs are discussing new backends
import pymc3 as pm

n_hidden = 20

with pm.Model() as nn_model:
  # Input -> Layer 1
  weights_1 = pm.Normal('w_1', mu=0, sd=1,
                        shape=(ann_input.shape[1], n_hidden),
                        testval=init_1)
  acts_1 = pm.Deterministic('activations_1',
                            tt.tanh(tt.dot(ann_input, weights_1)))

  # Layer 1 -> Layer 2
  weights_2 = pm.Normal('w_2', mu=0, sd=1,
                        shape=(n_hidden, n_hidden),
                        testval=init_2)
  acts_2 = pm.Deterministic('activations_2',
                            tt.tanh(tt.dot(acts_1, weights_2)))

  # Layer 2 -> Output Layer
  weights_out = pm.Normal('w_out', mu=0, sd=1,
                          shape=(n_hidden, ann_output.shape[1]),
                          testval=init_out)
  acts_out = pm.Deterministic('activations_out',
                              tt.nnet.softmax(tt.dot(acts_2, weights_out)))  # noqa

  # Define likelihood
  out = pm.Multinomial('likelihood', n=1, p=acts_out,
                       observed=ann_output)

with nn_model:
  s = theano.shared(pm.floatX(1.1))
  inference = pm.ADVI(cost_part_grad_scale=s)  # approximate inference done using ADVI
  approx = pm.fit(100000, method=inference)
  trace = approx.sample(5000)
