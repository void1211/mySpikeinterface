# Cars install
uv pip install git+https://github.com/void1211/CellActivityRecordingSimulator.git

n_chunk_length = 1000
RuntimeError: DARTsort execution failed: CUDA out of memory. Tried to allocate 5.37 GiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Of the allocated memory 17.03 GiB is allocated by PyTorch, and 689.68 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

n_chunk_length = 500
Traceback (most recent call last):
  File "c:\Users\tanaka-users\tlab\tlab_yasui\2025\my_spikeinterface\.venv\Lib\site-packages\spikeinterface\sorters\external\dartsort.py", line 266, in _run_from_folder
    result = dartsort.dartsort(
             ^^^^^^^^^^^^^^^^^^
  File "c:\Users\tanaka-users\tlab\tlab_yasui\2025\my_spikeinterface\.venv\Lib\site-packages\dartsort\main.py", line 80, in dartsort
    return _dartsort_impl(recording, output_dir, cfg, motion_est, None, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\tanaka-users\tlab\tlab_yasui\2025\my_spikeinterface\.venv\Lib\site-packages\dartsort\main.py", line 152, in _dartsort_impl
    sorting, _ = refine_clustering(
                 ^^^^^^^^^^^^^^^^^^
  File "c:\Users\tanaka-users\tlab\tlab_yasui\2025\my_spikeinterface\.venv\Lib\site-packages\dartsort\cluster\refine.py", line 39, in refine_clustering
    return gmm_refine(
           ^^^^^^^^^^^
  File "c:\Users\tanaka-users\tlab\tlab_yasui\2025\my_spikeinterface\.venv\Lib\site-packages\dartsort\cluster\refine.py", line 137, in gmm_refine
    res = gmm.tvi(final_split=intermediate_split)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\tanaka-users\tlab\tlab_yasui\2025\my_spikeinterface\.venv\Lib\site-packages\dartsort\cluster\gaussian_mixture.py", line 434, in tvi
    self.tmm = truncated_mixture.SpikeTruncatedMixtureModel(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\tanaka-users\tlab\tlab_yasui\2025\my_spikeinterface\.venv\Lib\site-packages\dartsort\cluster\truncated_mixture.py", line 98, in __init__
    self.processor = TruncatedExpectationProcessor(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\tanaka-users\tlab\tlab_yasui\2025\my_spikeinterface\.venv\Lib\site-packages\dartsort\cluster\truncated_mixture.py", line 543, in __init__
    self.initialize_fixed(noise, neighborhoods, pgeom=pgeom)
  File "c:\Users\tanaka-users\tlab\tlab_yasui\2025\my_spikeinterface\.venv\Lib\site-packages\dartsort\cluster\truncated_mixture.py", line 972, in initialize_fixed
    buf = torch.zeros(shp, device=device)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.42 GiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Of the allocated memory 17.25 GiB is allocated by PyTorch, and 91.86 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)




# Dartsort Settings

## DARTsortUserConfig:

### dredge_only: bool=False
Whether to stop after initial localization and motion tracking.

### matching_iterations :int=1

### n_jobs_cpu: int=0
Number of parallel workers to use when running on CPU. 0 means everything runs on the main thread.

### n_jobs_gpu: int=0
Number of parallel workers to use when running on GPU. 0 means everything runs on the main thread.

### device: str=None
The name of the PyTorch device to use. For example, 'cpu' or 'cuda' or 'cuda:1'. If unset, uses n_jobs_gpu of your CUDA GPUs if you have multiple, or else just the one, or your CPU.

### executor: srt="threading_unless_multigpu"

### chunk_length_samples: int=30,000

### work_in_tmpdir: bool=False

### tmpdir_parent: str|Path=None

### save_intermediate_labels: bool=False

### save_intermediate_features: bool=True

### save_final_features: bool=True

### ms_before: float=1.4
Length of time (ms) before trough (or peak) in waveform snippets. Default value corresponds to 42 samples at 30kHz. Default is 1.4.

### ms_after: flaot=2.6 + 0.1/3
Length of time (ms) after trough (or peak) in waveform snippets. Default value corresponds to 79 samples at 30kHz.

### alignment_ms: float=1.0
Time shift allowed when aligning events.

### initial_threshold: float=4.0
Threshold in standardized voltage units for initial detection; peaks or troughs larger than this value will be grabbed.

### matching_threshold: float=10.0 * 
Template matching threshold. If subtracting a template leads to at least this great of a decrease in the norm of the residual, that match will be used.

テンプレートを差し引いた際に、残差のノルムが少なくともこの値だけ減少する場合、そのマッチングが採用される。

### matching_fp_control: bool=False * 
Whether to control false positives in matching.

### denoiser_badness_factor: float=0.1
In initial detection, subtracting clean waveforms inferred by the NN denoiser need only decrease the residual norm squared by this multiple of the squared matching threshold to be accepted.

### temporal_pca_rank: int=8
Rank of global temporal PCA.

### feature_ms_before: float=0.75
As ms_before, but used only when computing PCA features in clustering.

### feature_ms_after: float=1.25
As ms_after, but used only when computing PCA features in clustering.

### subtraction_radius_um: float=200.0 * 
Radius of neighborhoods around spike events extracted when denoising and subtracting NN-denoised events.

### deduplication_radius_um: float=100.0 *
During initial detection, if two spike events occur at the same time within this radius, then the smaller of the two is ignored. But also all of the secondary channels of the big one, which is important.

### featurization_radius_um: float=100.0 * 
Radius around detection channel or template peak channel used to extract spike features for clustering.

### fit_radius_um: float=75.0
Extraction radius when fitting features like PCA; smaller than other radii to include less noise.

### localization_radius_um: float=100.0
Radius around main channel used when localizing spikes.

### density_bandwidth: float=5.0
Bandwidth for density estimation.

### interpolation_bandwidth: float=10.0
Bandwidth for interpolation.

### amplitude_scaling_stddev: float=0.1
Standard deviation for amplitude scaling.

### amplitude_scaling_limit: float=1.0
Limit for amplitude scaling.

### temporal_upsamples: int=4
Number of temporal upsamples.

### do_motion_estimation: bool=True
Set this to false if your data is super stable or already motion-corrected.

### rigid: bool=False
Use rigid registration and ignore the window parameters. 

### probe_boundary_padding_um: float=100.0 *
Padding around probe boundary in micrometers.

### spatial_bin_length_um: float=1.0
Length of spatial bins in micrometers.

### temporal_bin_length_s: float=1.0
Length of temporal bins in seconds.

### window_step_um: float=400.0
Step size for windows in micrometers.

### window_scale_um: float=45.0
Scale for windows in micrometers.

### window_margin_um: float=None
Margin for windows in micrometers.

### max_dt_s: float=1000.0
Maximum time difference in seconds.

### max_disp_um: float=None
Maximum displacement in micrometers.

### correlation_threshold: float=0.1
Threshold for correlations.

### min_amplitude: float=None
Minimum amplitude.

## DeveloperConfig:

@dataclass(frozen=True, kw_only=True, config=_strict_config)
class DeveloperConfig(DARTsortUserConfig):
    """Additional parameters for experiments. This API will never be stable."""

    initial_split_only: bool = True

    use_nn_in_subtraction: bool = True
    use_singlechan_templates: bool = False
    use_universal_templates: bool = False
    signal_rank: Annotated[int, Field(ge=0)] = 0
    truncated: bool = True
    overwrite_matching: bool = False

    criterion_threshold: float = 0.0
    criterion: Literal[
        "heldout_loglik", "heldout_elbo", "loglik", "elbo"
    ] = "heldout_elbo"
    merge_bimodality_threshold: float = 0.05
    n_refinement_iters: int = 3
    n_em_iters: int = 50
    channels_strategy: str = "count"
    hard_noise: bool = False

    initial_amp_feat: bool = True
    initial_pc_feats: int = 0
    initial_pc_scale: float = 2.5

    n_waveforms_fit: int = 20_000
    max_waveforms_fit: int = 50_000
    nn_denoiser_max_waveforms_fit: int = 250_000
    nn_denoiser_class_name: str = "SingleChannelWaveformDenoiser"
    nn_denoiser_pretrained_path: str | None = argfield(
        default=default_pretrained_path, arg_type=str_or_none
    )
    do_tpca_denoise: bool = True
    first_denoiser_thinning: float = 0.5

    gmm_max_spikes: Annotated[int, Field(gt=0)] = 4_000_000
    gmm_val_proportion: Annotated[float, Field(gt=0)] = 0.25
    gmm_split_decision_algorithm: str = "brute"
    gmm_merge_decision_algorithm: str = "brute"
    prior_pseudocount: float = 5.0
    cov_kind: str = "factorizednoise"
    interpolation_method: str = "kriging"
    extrapolation_method: str | None = argfield(default="kernel", arg_type=str_or_none)
    interpolation_kernel: str = "thinplate"
    interpolation_rq_alpha: float = 0.5
    interpolation_degree: int = 1
    glasso_alpha: float | int | None = argfield(default=None, arg_type=int_or_float_or_none)
    laplace_ard: bool = True
    core_radius: float = 35.0
