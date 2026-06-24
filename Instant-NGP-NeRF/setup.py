from setuptools import setup, find_packages

setup(
    name=nerf_teaching,
    version=1.0.0,
    author=The Authors,
    description=Instant-NGP accelerated NeRF for 3D animation teaching scene reconstruction,
    packages=find_packages(where=src),
    package_dir={ src},
    install_requires=[
        torch=2.0.0,
        numpy=1.24.0,
        scipy=1.10.0,
        opencv-python=4.8.0,
        pillow=9.5.0,
        tqdm=4.65.0,
        pyyaml=6.0,
        scikit-image=0.20.0,
        lpips=0.1.4,
    ],
    python_requires==3.10,
    entry_points={
        console_scripts [
            nerf_train=src.train_nerfmain,
            nerf_distill=src.distill_gaussianmain,
            nerf_vr=src.vr_interactmain,
            nerf_eval=src.run_evaluationmain,
        ],
    },
)