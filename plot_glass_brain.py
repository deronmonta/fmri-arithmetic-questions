from nilearn import datasets
from nilearn import plotting
from nilearn.datasets import fetch_neurovault

haxby_dataset = datasets.fetch_haxby()
print(haxby_dataset)
# print basic information on the dataset
print('First subject anatomical nifti image (3D) is at: %s' %
      haxby_dataset.anat[0])
print('First subject functional nifti image (4D) is at: %s' %
      haxby_dataset.func[0])  # 4D data

haxby_anat_filename = haxby_dataset.anat[0]
haxby_mask_filename = haxby_dataset.mask_vt[0]
haxby_func_filename = haxby_dataset.func[0]

# one motor contrast map from NeuroVault
motor_images = datasets.fetch_neurovault_motor_task()
stat_img = motor_images.images[0]


plotting.plot_stat_map(stat_img,
                       threshold=3, title="plot_stat_map",
                       cut_coords=[36, -27, 66])
plotting.plot_glass_brain(stat_img, title='plot_glass_brain',
                          threshold=3,output_file='nilearn.png')
