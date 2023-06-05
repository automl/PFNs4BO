# PFNs4BO: In-Context Learning for Bayesian Optimization ([Arxiv](https://arxiv.org/abs/2305.17535))


## Install
```bash
pip install pfns4bo
```


## Use our trained models
The models used for our final setup can be found in `pfns4bo/final_models`.


### Use on discrete benchmarks (also in `Tutorial_Discrete_Interface.ipynb`)
To use our model on discrete benchmarks, we recommend using our HPO-B interface `pfns4bo.scripts.acquisition_functions.TransformerBOMethod`.

We called this interface like this for the eval's on HPO-B:

```python
import pfns4bo
import torch
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
from pfns4bo.scripts.tune_input_warping import fit_input_warping

# For HEBO+ 
model_path = pfns4bo.hebo_plus_model
# For BNN
# model_path = pfns4bo.bnn_model

# for correctly specified search spaces (e.g. correctly applied log transformations)
pfn_bo = TransformerBOMethod(torch.load(model_path), device='cpu:0')

# for mis-specified search spaces
pfn_bo = TransformerBOMethod(torch.load(model_path), fit_encoder=fit_input_warping, device='cpu:0')
```
The interface expects all features to be normalized to a [0,1] range and all features have to be scalars/floats.
```python
import numpy as np

X_obs = np.random.rand(4,1) # of shape num_examples x num_features of scalars
y_obs = np.abs(X_obs[:,0] - .5) * 2. # of shape num_examples
X_pen = np.linspace(0,1,100)[:,None] # of shape num_examples_pending x num_features

assert (X_obs <= 1).all() and (X_obs >= 0).all() and (X_pen <= 1).all() and (X_pen >= 0).all()
index_to_be_queried_next_in_pending = pfn_bo.observe_and_suggest(X_obs, y_obs, X_pen, return_actual_ei=False)
```

To use a different acquisition function than EI, you simply pass `acq_function='pi` or `acq_function='ucb'`.

To explore the EI's of the model you can do
```python
index_to_be_queried_next_in_pending, eis = pfn_bo.observe_and_suggest(X_obs, y_obs, X_pen, return_actual_ei=True)
```
The `eis` are the EI's of the model, i.e. the EI's of the model's predictive distribution for each `X_pen`.


### Use on continuous benchmarks (also in `Tutorial_Continuous_Interface.ipynb`)
To use our model on continuous setups, we recommend using the interface in `pfns4bo/pfn_bo_bayesmark.py`.
This is a standard BayesMark interface.
The calls to this interface used in our BayesMark experiments are given in `pfns4bo/config.json`.


### Use user priors (also in `Tutorial_Discrete_Interface.ipynb`)
The model is at 'pfns4bo/final_models/hebo_morebudget_9_unused_features_3_userpriorperdim2_8.pt'.
It can be used with both interfaces, but we only used it with the discrete interface for our experiments.
This is the setup we used for our PD-1 experiments for example.
```python
import torch
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod

# the order of hps in our benchmark is 'lr_decay_factor', 'lr_initial', 'lr_power', 'opt_momentum', 'epoch', 'activation'
pfn_bo = TransformerBOMethod(torch.load(pfns4bo.hebo_plus_userprior_model),
    style=
        torch.tensor([
            .5, 3/4, 4/4,  # feature 1 has .5 prob to the prior where all max's lie in [.75,1.], 1-.5=.5 prob to the standard prior
            .25, 2/4, 3/4, # feature 2 has .25 prob is given to the prior where all max's lie in [.5,.75]...
            .1, 3/4, 4/4,
            0., 0/1, 1/1,
            .5, 3/4, 4/4,
            .5, 4/5, 5/5,
        ]).view(1,-1)
)
```
All bounds must have the form `(k/n,(k+1)/n)` for `n in {1,2,3,4,5}` and `k in set(range(k))`.
Other bounds won't give an error, but very likely worse performance.
The PFN was only trained for these bounds.


## Train your own models (also in `Tutorial_Training.ipynb`)
To train we recommend installing the package locally after cloning it,
with `pip install -e .`.


Now you simply need to call `train.train`.
We give all necessary code. The most important bits are in the `priors` dir, e.g. `hebo_prior`, it stores the priors
with which we train our models.

You can train this model on 8 GPUs using `torchrun` or `submitit`
```python
import torch
from pfns4bo import priors, encoders, utils, bar_distribution, train
from ConfigSpace import hyperparameters as CSH

config_heboplus = {
     'priordataloader_class': priors.get_batch_to_dataloader(
         priors.get_batch_sequence(
             priors.hebo_prior.get_batch,
             priors.utils.sample_num_feaetures_get_batch,
         )
     ),
     'encoder_generator': encoders.get_normalized_uniform_encoder(encoders.get_variable_num_features_encoder(encoders.Linear)),
     'emsize': 512,
     'nhead': 4,
     'warmup_epochs': 5,
     'y_encoder_generator': encoders.Linear,
     'batch_size': 128,
     'scheduler': utils.get_cosine_schedule_with_warmup,
     'extra_prior_kwargs_dict': {'num_features': 18,
      'hyperparameters': {
       'lengthscale_concentration': 1.2106559584074301,
       'lengthscale_rate': 1.5212245992840594,
       'outputscale_concentration': 0.8452312502679863,
       'outputscale_rate': 0.3993553245745406,
       'add_linear_kernel': False,
       'power_normalization': False,
       'hebo_warping': False,
       'unused_feature_likelihood': 0.3,
       'observation_noise': True}},
     'epochs': 50,
     'lr': 0.0001,
     'bptt': 60,
     'single_eval_pos_gen': utils.get_uniform_single_eval_pos_sampler(50, min_len=1), #<function utils.get_uniform_single_eval_pos_sampler.<locals>.<lambda>()>,
     'aggregate_k_gradients': 2,
     'nhid': 1024,
     'steps_per_epoch': 1024,
     'weight_decay': 0.0,
     'train_mixed_precision': False,
     'efficient_eval_masking': True,
     'nlayers': 12}


config_heboplus_userpriors = {**config_heboplus,
    'priordataloader_class': priors.get_batch_to_dataloader(
                              priors.get_batch_sequence(
                                  priors.hebo_prior.get_batch,
                                  priors.condition_on_area_of_opt.get_batch,
                                  priors.utils.sample_num_feaetures_get_batch
                              )),
    'style_encoder_generator': encoders.get_normalized_uniform_encoder(encoders.get_variable_num_features_encoder(encoders.Linear))
}

config_bnn = {'priordataloader_class': priors.get_batch_to_dataloader(
         priors.get_batch_sequence(
             priors.simple_mlp.get_batch,
             priors.input_warping.get_batch,
             priors.utils.sample_num_feaetures_get_batch,
         )
     ),
     'encoder_generator': encoders.get_normalized_uniform_encoder(encoders.get_variable_num_features_encoder(encoders.Linear)),
     'emsize': 512,
     'nhead': 4,
     'warmup_epochs': 5,
     'y_encoder_generator': encoders.Linear,
     'batch_size': 128,
     'scheduler': utils.get_cosine_schedule_with_warmup,
     'extra_prior_kwargs_dict': {'num_features': 18,
      'hyperparameters': {'mlp_num_layers': CSH.UniformIntegerHyperparameter('mlp_num_layers', 8, 15),
       'mlp_num_hidden': CSH.UniformIntegerHyperparameter('mlp_num_hidden', 36, 150),
       'mlp_init_std': CSH.UniformFloatHyperparameter('mlp_init_std',0.08896049884896237, 0.1928554813280186),
       'mlp_sparseness': 0.1449806273312999,
       'mlp_input_sampling': 'uniform',
       'mlp_output_noise': CSH.UniformFloatHyperparameter('mlp_output_noise', 0.00035983014290491186, 0.0013416342770574585),
       'mlp_noisy_targets': True,
       'mlp_preactivation_noise_std': CSH.UniformFloatHyperparameter('mlp_preactivation_noise_std',0.0003145707276259681, 0.0013753183831259406),
       'input_warping_c1_std': 0.9759720822120248,
       'input_warping_c0_std': 0.8002534583197192,
       'num_hyperparameter_samples_per_batch': 16}
                                 },
     'epochs': 50,
     'lr': 0.0001,
     'bptt': 60,
     'single_eval_pos_gen': utils.get_uniform_single_eval_pos_sampler(50, min_len=1), 
     'aggregate_k_gradients': 1,
     'nhid': 1024,
     'steps_per_epoch': 1024,
     'weight_decay': 0.0,
     'train_mixed_precision': True,
     'efficient_eval_masking': True,
}


# now let's add the criterions, where we decide the border positions based on the prior
def get_ys(config):
    bs = 128
    all_targets = []
    for num_hps in [2,8,12]: # a few different samples in case the number of features makes a difference in y dist
        b = config['priordataloader_class'].get_batch_method(bs,1000,num_hps,epoch=0,device='cuda:0',
                                                              hyperparameters=
                                                              {**config['extra_prior_kwargs_dict']['hyperparameters'],
                                                               'num_hyperparameter_samples_per_batch': -1,})
        all_targets.append(b.target_y.flatten())
    return torch.cat(all_targets,0)

def add_criterion(config):
    return {**config, 'criterion': bar_distribution.FullSupportBarDistribution(
        bar_distribution.get_bucket_limits(1000,ys=get_ys(config).cpu())
    )}

# Now let's train either with
train.train(**add_criterion(config_heboplus))
# or
train.train(**add_criterion(config_heboplus_userpriors))
# or
train.train(**add_criterion(config_bnn))
```

### Problem Handling
**Out of memory during inference**: It might be fixed by changing `max_dataset_size=10_000` to something smaller on either interface.

### You can cite our work with
```
@article{muller2023pfns,
  title={PFNs4BO: In-Context Learning for Bayesian Optimization},
  author={M{\"u}ller, Samuel and Feurer, Matthias and Hollmann, Noah and Hutter, Frank},
  journal={arXiv preprint arXiv:2305.17535},
  year={2023}
}
```
