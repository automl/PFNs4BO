import os

def unzip_models():
    pfns4bo_dir = os.path.dirname(__file__)
    model_names = ['hebo_morebudget_9_unused_features_3_userpriorperdim2_8.pt',
                   'model_sampled_warp_simple_mlp_for_hpob_46.pt',
                   'model_hebo_morebudget_9_unused_features_3.pt',]
    model_path = 'final_models'

    print('Hello')
    for name in model_names:
        if not os.path.exists(os.path.join(pfns4bo_dir, model_path, name)):
            if os.path.exists(os.path.join(pfns4bo_dir, model_path, name + '.gz')):
                print("Unzipping", name)
                os.system(f"gzip -dk {os.path.join(pfns4bo_dir, model_path, name + '.gz')}")
            else:
                print("Failed to find", os.path.abspath(os.path.join(pfns4bo_dir, model_path, name + '.gz')))
