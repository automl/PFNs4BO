import os

model_path = 'final_models'

def prepare_models():
    pfns4bo_dir = os.path.dirname(__file__)
    model_names = ['hebo_morebudget_9_unused_features_3_userpriorperdim2_8.pt',
                   'model_sampled_warp_simple_mlp_for_hpob_46.pt',
                   'model_hebo_morebudget_9_unused_features_3.pt',]

    for name in model_names:
        weights_path = os.path.join(pfns4bo_dir, model_path, name)
        compressed_weights_path = os.path.join(pfns4bo_dir, model_path, name + '.gz')
        if not os.path.exists(weights_path):
            if not os.path.exists(compressed_weights_path):
                print("Downloading", os.path.abspath(compressed_weights_path))
                import requests
                url = f'https://github.com/automl/PFNs4BO/raw/main/pfns4bo/final_models/{name + ".gz"}'
                r = requests.get(url, allow_redirects=True)
                os.makedirs(os.path.dirname(compressed_weights_path), exist_ok=True)
                with open(compressed_weights_path, 'wb') as f:
                    f.write(r.content)
            if os.path.exists(compressed_weights_path):
                print("Unzipping", name)
                os.system(f"gzip -dk {compressed_weights_path}")
            else:
                print("Failed to find", compressed_weights_path)
                print("Make sure you have an internet connection to download the model automatically..")
        if os.path.exists(weights_path):
            print("Successfully located model at", weights_path)


model_dict = {
    'hebo_plus_userprior_model': os.path.join(os.path.dirname(__file__),model_path,
                                              'hebo_morebudget_9_unused_features_3_userpriorperdim2_8.pt'),
    'hebo_plus_model': os.path.join(os.path.dirname(__file__),model_path,
                                    'model_hebo_morebudget_9_unused_features_3.pt'),
    'bnn_model': os.path.join(os.path.dirname(__file__),model_path,'model_sampled_warp_simple_mlp_for_hpob_46.pt')
}


def __getattr__(name):
    if name in model_dict:
        if not os.path.exists(model_dict[name]):
            print("Can't find", os.path.abspath(model_dict[name]), "thus unzipping/downloading models now.")
            print("This might take a while..")
            prepare_models()
        return model_dict[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

